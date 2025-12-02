from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from packaging.version import Version
from scipy.sparse import issparse

from .datasets import MofaFlexDataset

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from . import MOFAFLEX


MOFACompatOption = Literal["full", "modelonly"] | bool


def _save_mofa_data(adata, group_name, view_name, data_grp, dset_kwargs):
    arr = adata.X if not issparse(adata.X) else adata.X.toarray()
    arr = align_local_array_to_global(arr, group_name, view_name, align_to="samples", axis=0, fill_value=np.nan)  # noqa F821
    arr = align_local_array_to_global(arr, group_name, view_name, align_to="features", axis=1, fill_value=np.nan)  # noqa F821
    data_grp.create_dataset(f"{view_name}/{group_name}", data=arr, **dset_kwargs)


def save_model(
    model_state,
    path: str | Path,
    mofa_compat: bool = False,
    model: MOFAFLEX | None = None,
    data: MofaFlexDataset | None = None,
    intercepts: dict[str, dict[str, NDArray[np.number]]] | None = None,
):
    """Save a MOFA-FLEX model to an HDF5 file.

    Saves both the model state and parameters, with optional MOFA-compatible format.

    Args:
        model_state: The internal state of the model. Should be compatible with `anndata.io.write_elem`.
        path: File path where to save the model.
        mofa_compat: If True, saves additional data in MOFA-compatible format.
        model: The MOFA-FLEX model to save. Only needed for `mofa_compat=True`.
        data: The input data. Only needed for `mofa_compat=True`.
        intercepts: The data intercepts. Only needed for `mofa_compat=True`.
    """
    if mofa_compat and model is None:
        raise ValueError("Need a MOFAFLEX object if saving in MOFA compatibility mode.")
    if (mofa_compat is True or mofa_compat == "full") and (data is None or intercepts is None):
        raise ValueError("Need both data and intercepts if saving data in MOFA compatibility mode.")

    from .. import __version__

    dset_kwargs = {"compression": "gzip", "compression_opts": 9}

    path = Path(path)
    if path.exists():
        logger.warning(f"{path} already exists, overwriting")
    with h5py.File(path, "w") as f:
        with ad.settings.override(allow_write_nullable_strings=True):
            ad.io.write_elem(
                f,
                "mofaflex",
                model_state,
                dataset_kwargs={} if Version(ad.__version__) < Version("0.11.2") else dset_kwargs,
            )  # https://github.com/h5py/h5py/issues/2525

        f["mofaflex"].attrs["version"] = __version__

        if mofa_compat:
            # save MOFA-compatible output
            # This currently uses some private model attributes that are not part of the public API.
            # Not the cleanest design, but otoh I don't think these things should be part of our
            # API at the moment.'
            f.create_dataset("groups/groups", data=model.group_names.astype("O"), **dset_kwargs)
            f.create_dataset("views/views", data=model.view_names.astype("O"), **dset_kwargs)

            samples_grp = f.create_group("samples")
            for group_name, group_samples in model.sample_names.items():
                samples_grp.create_dataset(group_name, data=group_samples, **dset_kwargs)

            features_grp = f.create_group("features")
            for view_name, view_features in model.feature_names.items():
                features_grp.create_dataset(view_name, data=view_features, **dset_kwargs)

            if hasattr(model, "covariates"):
                covar_names = None
                if len(model.covariates_names) == 1:
                    covar_names = next(model.covariates_names.values())
                elif len(model.covariates_names) > 1:
                    groups = list(model.covariates_names.keys())
                    lengths = [len(g) for g in model.covariates_names.values()]
                    refidx = np.argmax(lengths)
                    if groups[refidx] in model.covariates_names:
                        ref = set(model.covariates_names[groups[refidx]])
                        if all(set(gc) <= ref for gc in model.covariates_names.values()):
                            covar_names = model.covariates_names[groups[refidx]]
                if covar_names is None:
                    maxlen = max(c.shape[1] for c in model.covariates.values())
                    covar_names = [f"covar_{i}" for i in range(maxlen)]

                f.create_dataset("covariates/covariates", data=covar_names)

                cov_grp = f.create_group("cov_samples")
                for g_name, covars in model.covariates.items():
                    cov_grp.create_dataset(g_name, data=covars, **dset_kwargs)

                warped_covs = model.warped_covariates
                if warped_covs is not None:
                    cov_grp = f.create_group("cov_samples_transformed")
                    for g_name, covars in warped_covs.items():
                        cov_grp.create_dataset(g_name, data=covars, **dset_kwargs)

            samples_meta_grp = f.create_group("samples_metadata")
            for group_name in model.group_names:
                cgrp = samples_meta_grp.create_group(group_name)
                metadata = model._metadata[group_name]
                cntr = Counter()
                for v in metadata.values():
                    cntr.update(v.columns)
                df = pd.concat(
                    (
                        df.rename(columns={col: f"{view_name}:{col}" for col in df.columns if cntr[col] > 1})
                        for view_name, df in metadata.items()
                    ),
                    axis=1,
                ).reset_index()
                for i in range(df.shape[1]):
                    col = df.iloc[:, i]
                    cvals = col.to_numpy()
                    if cvals.dtype == "O":
                        cvals[col.isna()] = ""  # np.isnan doesn't work on object arrays
                    cgrp.create_dataset(col.name, data=cvals, **dset_kwargs)

            if mofa_compat == "full":
                intercept_grp = f.create_group("intercepts")
                for group_name, gintercepts in intercepts.items():
                    for view_name, intercept in gintercepts.items():
                        cgrp = intercept_grp.require_group(view_name)
                        cgrp.create_dataset(group_name, data=intercept, **dset_kwargs)

                data_grp = f.create_group("data")
                data.apply(_save_mofa_data, data_grp=data_grp, dset_kwargs=dset_kwargs)

            exp_grp = f.create_group("expectations")
            factor_grp = exp_grp.create_group("Z")
            for group_name, factors in model.get_factors(return_type="numpy", ordered=False).items():
                factor_grp.create_dataset(group_name, data=factors.T, **dset_kwargs)

            weight_grp = exp_grp.create_group("W")
            for view_name, weights in model.get_weights(return_type="numpy", ordered=False).items():
                weight_grp.create_dataset(view_name, data=weights, **dset_kwargs)

            # save Sigma?

            model_opts_grp = f.create_group("model_options")
            model_opts_grp.create_dataset(
                "likelihoods",
                data=[str(model._model_opts.likelihoods[v]).lower() for v in model.view_names],
                **dset_kwargs,
            )
            model_opts_grp.create_dataset(
                "spikeslab_factors",
                data=any(p.__class__.__name__ == "SpikeSlab" for p in model._model_opts.factor_prior),
            )
            model_opts_grp.create_dataset(
                "spikeslab_weights",
                data=any(p.__class__.__name__ == "SpikeSlab" for p in model._model_opts.weight_prior),
            )
            # ARD used unconditionally in SnS prior
            model_opts_grp.create_dataset(
                "ard_factors", data=any(p.__class__.__name__ == "SpikeSlab" for p in model._model_opts.factor_prior)
            )
            model_opts_grp.create_dataset(
                "ard_weights", data=any(p.__class__.__name__ == "SpikeSlab" for p in model._model_opts.weight_prior)
            )

            train_opts_grp = f.create_group("training_opts")
            train_opts_grp.create_dataset("maxiter", data=model._train_opts.max_epochs)
            train_opts_grp.create_dataset("freqELBO", data=1)
            train_opts_grp.create_dataset("start_elbo", data=0)
            train_opts_grp.create_dataset("gpu_mode", data=model._train_opts.device.type != "cpu")
            train_opts_grp.create_dataset("stochastic", data=True)

            if hasattr(model, "covariates"):
                smooth_opts_grp = f.create_group("smooth_opts")
                smooth_opts_grp.create_dataset("scale_cov", data=model._gp_opts.independent_lengthscales)
                smooth_opts_grp.create_dataset("start_opt", data=0)
                smooth_opts_grp.create_dataset("opt_freq", data=1)
                smooth_opts_grp.create_dataset("sparseGP", data=True)
                smooth_opts_grp.create_dataset("warping_freq", data=model._gp_opts.warp_interval)
                if model._gp_opts.warp_reference_group is not None:
                    smooth_opts_grp.create_dataset("warping_ref", data=model._gp_opts.warp_reference_group)
                smooth_opts_grp.create_dataset("warping_open_begin", data=model._gp_opts.warp_open_begin)
                smooth_opts_grp.create_dataset("warping_open_end", data=model._gp_opts.warp_open_end)
                smooth_opts_grp.create_dataset("model_groups", data=True)

            varexp_grp = f.create_group("variance_explained")
            varexp_factor_grp = varexp_grp.create_group("r2_per_factor")
            for group_name, df in model.get_r2(total=False, ordered=True).items():
                varexp_factor_grp.create_dataset(group_name, data=df.to_numpy().T * 100, **dset_kwargs)

            varexp_total_grp = varexp_grp.create_group("r2_total")
            view_names = np.asarray(model.view_names)
            for group_name, df in model.get_r2(total=True).items():
                varexp_total_grp.create_dataset(
                    group_name, data=df[view_names[np.isin(view_names, df.index)]] * 100, **dset_kwargs
                )

            train_stats_grp = f.create_group("training_stats")
            train_stats_grp.create_dataset("elbo", data=model.training_loss, **dset_kwargs)
            if hasattr(model, "covariates"):
                train_stats_grp.create_dataset("length_scales", data=model.gp_lengthscale, **dset_kwargs)
                train_stats_grp.create_dataset("scales", data=model.gp_scale, **dset_kwargs)
                train_stats_grp.create_dataset("Kg", data=model.gp_group_correlation, **dset_kwargs)


def load_model(path: str | Path):
    """Load a MOFA-FLEX model from an HDF5 file.

    Args:
        path: Path to the HDF5 file containing the saved model.
        map_location: Optional device specification for loading the model.

    Returns:
        The loaded MOFA-FLEX model.
    """
    from .. import __version__

    path = Path(path)
    with h5py.File(path, "r") as f:
        mofaflexgrp = f["mofaflex"]
        if mofaflexgrp.attrs["version"] != __version__:
            logger.warning(
                "The stored model was created with a different version of MOFA-FLEX. Some features may not work."
            )
        state = ad.io.read_elem(mofaflexgrp)

    return state
