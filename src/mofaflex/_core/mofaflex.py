import logging
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from functools import wraps
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyro
import torch
from anndata import AnnData
from mudata import MuData
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
from torch.utils.data import DataLoader, default_convert
from torch.utils.data._utils.collate import collate  # this is documented, so presumably part of the public API
from tqdm.auto import tqdm
from tqdm.notebook import tqdm_notebook

from .. import pl
from .api.likelihoods import Likelihood as APILikelihood
from .datasets import MofaFlexBatchSampler, MofaFlexDataset, StackDataset
from .io import load_model, save_model
from .likelihoods import LikelihoodType
from .model import MofaFlexModel
from .terms import Term, TermWrapper
from .training import EarlyStopper
from .utils import default_torch_device, filter_constant_features, sample_all_data_as_one_batch

_logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class _Options:
    def asdict(self):
        # avoid the deepcopy done by dataclasses.asdict
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __post_init__(self):
        # after an HDF5 roundtrip, these are numpy scalars, which PyTorch doesn't handle well'
        for f in fields(self):
            if f.type in (float, int, bool):
                setattr(self, f.name, f.type(getattr(self, f.name)))


@dataclass(kw_only=True)
class _DataOptions(_Options):
    group_by: str | Sequence[str] | None
    layer: Mapping[str, str | None] | Mapping[str, Mapping[str, str | None]] | str | None
    use_obs: Literal["union", "intersection"] | None
    use_var: Literal["union", "intersection"] | None
    subset_var: str | None
    remove_constant_features: bool


@dataclass(kw_only=True)
class _TrainingOptions(_Options):
    device: str | torch.device
    batch_size: int
    max_epochs: int
    n_particles: int
    lr: float
    early_stopper_patience: int
    save_path: Path | str | None
    seed: int | None
    num_workers: int
    pin_memory: bool

    def __post_init__(self):
        super().__post_init__()
        self.device = torch.device(self.device)


class MOFAFLEX:
    """The MOFA-FLEX model.

    This class is not meant to be instantiated by the user. Rather, it is created by instantiating a :mod:`term <.terms>`.
    """

    def __init__(self, **kwargs: Term):
        self._terms = kwargs

    def __add__(self, other: "MOFAFLEX"):
        if not isinstance(other, __class__):
            return NotImplemented
        if hasattr(self, "_model") or hasattr(other, "_model"):
            raise ValueError("Cannot add terms to an already trained model.")
        if len(intersection := self._terms.keys() & other._terms.keys()) > 0:
            raise ValueError(
                f"Operands must have unique term names, but terms {', '.join(intersection)} were found in both operands."
            )
        return __class__(**self._terms, **other._terms)

    def __dir__(self):
        sdir = super().__dir__()
        if hasattr(self, "_wrapped_terms") and self.n_terms == 1:
            return chain(sdir, next(iter(self._wrapped_terms.values())).__dir__(forward=False))
        else:
            return sdir

    def __getattr__(self, name):
        if "_wrapped_terms" in self.__dict__ and self.n_terms == 1:
            return next(iter(self._wrapped_terms.values())).__getattr__(name, forward=False)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'", name=name, obj=self)

    def _make_dataset(self, data: MuData | Mapping[str, Mapping[str, AnnData]]) -> MofaFlexDataset:
        return MofaFlexDataset(
            data,
            layer=self._data_opts.layer,
            group_by=self._data_opts.group_by,
            use_obs=self._data_opts.use_obs,
            use_var=self._data_opts.use_var,
            subset_var=self._data_opts.subset_var,
        )

    def _check_trained(func=None, *, is_trained=True):
        def wrapperwrapper(func):
            @wraps(func)
            def wrapper(self, *args, **kwargs):
                if not getattr(self, "_no_check_trained_", False):
                    if is_trained and not hasattr(self, "_model"):
                        raise RuntimeError("The model is not yet trained.")
                    elif not is_trained and hasattr(self, "_model"):
                        raise RuntimeError("The model is already trained.")
                return func(self, *args, **kwargs)

            return wrapper

        if func is None:
            return lambda func: wrapperwrapper(func)
        else:
            return wrapperwrapper(func)

    @contextmanager
    def _no_check_trained(self):
        self._no_check_trained_ = True
        yield
        self._no_check_trained_ = False

    @property
    @_check_trained
    def group_names(self) -> npt.NDArray[str]:
        """Group names."""
        return self._group_names

    @property
    @_check_trained
    def n_groups(self) -> int:
        """Number of groups."""
        return len(self.group_names)

    @property
    @_check_trained
    def view_names(self) -> npt.NDArray[str]:
        """View names."""
        return self._view_names

    @property
    @_check_trained
    def n_views(self) -> int:
        """Number of views."""
        return len(self.view_names)

    @property
    @_check_trained
    def feature_names(self) -> Mapping[str, npt.NDArray[str]]:
        """Feature names for each view."""
        return MappingProxyType(self._feature_names)

    @property
    @_check_trained
    def n_features(self) -> dict[str, int]:
        """Number of features in each view."""
        return {k: len(v) for k, v in self.feature_names.items()}

    @property
    @_check_trained
    def n_features_total(self) -> int:
        """Total number of features."""
        return sum(self.n_features.values())

    @property
    @_check_trained
    def sample_names(self) -> Mapping[str, npt.NDArray[str]]:
        """Sample names for each group."""
        return MappingProxyType(self._sample_names)

    @property
    @_check_trained
    def n_samples(self) -> dict[str, int]:
        """Number of samples in each group."""
        return {k: len(v) for k, v in self.sample_names.items()}

    @property
    @_check_trained
    def n_samples_total(self) -> int:
        """Total number of samples."""
        return sum(self.n_samples.values())

    @property
    @_check_trained
    def training_loss(self) -> npt.NDArray[np.float32]:
        """Total loss (negative ELBO) for each training epoch."""
        return self._train_loss_elbo

    @property
    @_check_trained
    def terms(self) -> Mapping[str, Term]:
        """The additive terms."""
        return MappingProxyType(self._wrapped_terms)

    @property
    def n_terms(self) -> int:
        """Number of additive terms."""
        try:
            return len(self._wrapped_terms)
        except AttributeError:
            return len(self._terms)

    def _init_api(self):
        self._wrapped_terms = {name: TermWrapper(self, term) for name, term in self._model.terms.items()}

    @_check_trained(is_trained=False)
    def fit(
        self,
        data: MuData | Mapping[str, Mapping[str, AnnData]] | AnnData,
        *,
        likelihoods: Mapping[str, LikelihoodType | APILikelihood] | LikelihoodType | APILikelihood | None = None,
        group_by: str | Sequence[str] | None = None,
        layer: Mapping[str, str | None] | Mapping[str, Mapping[str, str | None]] | str | None = None,
        use_obs: Literal["union", "intersection"] = "union",
        use_var: Literal["union", "intersection"] = "union",
        subset_var: str | None = "highly_variable",
        plot_data_overview: bool = True,
        remove_constant_features: bool = True,
        device: str | torch.device = "cuda",
        batch_size: int = 0,
        max_epochs: int = 10_000,
        lr: float = 0.001,
        early_stopper_patience: int = 100,
        save_path: Path | str | None = None,
        seed: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        n_particles: int = 1,
    ):
        """Fit the model using the provided data.

        Args:
            data: can be any of:

                - MuData object
                - Nested dict with group names as keys, view names as subkeys and AnnData objects as values
                  (incompatible with `.group_by`)

            likelihoods: Data likelihoods for each view (if dict) or for all views (if str or Likelihood).
                Inferred automatically if None.
            group_by: Columns of `.obs` in :class:`MuData<mudata.MuData>` or :class:`AnnData<anndata.AnnData>` objects to group
                data by. Ignored if the input data is not a :class:`MuData<mudata.MuData>` or :class:`AnnData<anndata.AnnData>` object.
            layer: Which layer to use. If `None`, the `.X` element will be used. If `str`, the same layer will be used for
                all groups and views. If a dict of strings, the keys must correspond to view names and the values to layers.
                If a nested dict, different layers can be used for each combination of group and view. The last format is
                only accepted if the data is a nested dictionary of :class:`AnnData<anndata.AnnData>` objects.
            use_obs: How to align observations across views. Ignored if the data is not a nested dict of
                :class:`AnnData<anndata.AnnData>` objects.
            use_var: How to align variables across groups. Ignored if the data is not a nested dict of
                :class:`AnnData<anndata.AnnData>` objects.
            subset_var: `.var` column with boolean values to select features.
            plot_data_overview: Plot data overview.
            remove_constant_features: Remove constant features from the data.
            device: Device to run training on.
            batch_size: Batch size.
            max_epochs: Maximum number of training epochs.
            lr: Learning rate.
            early_stopper_patience: Number of steps without relevant improvement to stop training.
            save_path: Path to save model.
            seed: Seed for the pseudorandom number generator.
            num_workers: Number of data loader workers.
            pin_memory: Whether to use pinned memory in the data loader.
            n_particles: Number of particles for ELBO estimation.
        """
        self._data_opts = _DataOptions(
            group_by=group_by,
            layer=layer,
            use_obs=use_obs,
            use_var=use_var,
            subset_var=subset_var,
            remove_constant_features=remove_constant_features,
        )
        self._train_opts = _TrainingOptions(
            device=device,
            batch_size=batch_size,
            max_epochs=max_epochs,
            n_particles=n_particles,
            lr=lr,
            early_stopper_patience=early_stopper_patience,
            save_path=save_path,
            seed=seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        data = self._make_dataset(data)
        if self._data_opts.remove_constant_features:
            filter_constant_features(data)

        self._metadata = data.get_obs()
        self._view_names = data.view_names
        self._group_names = data.group_names
        self._sample_names = data.sample_names
        self._feature_names = data.feature_names

        if self._train_opts.seed is not None:
            try:
                self._train_opts.seed = int(self._train_opts.seed)
            except ValueError:
                _logger.warning(f"Could not convert `{self._train_opts.seed}` to integer.")
                self._train_opts.seed = None

        if self._train_opts.seed is None:
            self._train_opts.seed = int(time.strftime("%y%m%d%H%M"))

        self._train_opts.device = default_torch_device(self._train_opts.device)
        with self._no_check_trained():
            if self._train_opts.batch_size is None or not (0 < self._train_opts.batch_size <= self.n_samples_total):
                self._train_opts.batch_size = self.n_samples_total

            if plot_data_overview:
                pl.overview(data).show()

            pyro.set_rng_seed(self._train_opts.seed)
            model = MofaFlexModel(terms=self._terms, likelihoods=likelihoods).to(self._train_opts.device)

            n_iterations = int(self._train_opts.max_epochs * (self.n_samples_total // self._train_opts.batch_size))
            gamma = 0.1
            lrd = gamma ** (1 / n_iterations)

            datasets = {"data": data}
            if (termdsets := model.get_datasets(data)) is not None:
                datasets.update(termdsets)

            pyro.enable_validation(True)
            pyro.clear_param_store()

            singlebatch = self._train_opts.batch_size >= max(self.n_samples.values())
            collate_fn_map = {
                torch.Tensor: lambda x, **kwargs: x[0].to(self._train_opts.device, non_blocking=True),
                slice: lambda x, **kwargs: x[0],
            }
            dataset = StackDataset(**datasets)
            if singlebatch:
                batch = collate(
                    (default_convert(dataset.__getitems__(sample_all_data_as_one_batch(data))),),
                    collate_fn_map=collate_fn_map,
                )
                batchdata = batch.pop("data")
            else:
                loader = DataLoader(
                    dataset,
                    batch_sampler=MofaFlexBatchSampler(
                        data.n_samples, self._train_opts.batch_size, False, generator=torch.default_generator
                    ),
                    collate_fn=default_convert,
                    num_workers=self._train_opts.num_workers,
                    pin_memory=self._train_opts.pin_memory,
                    persistent_workers=self._train_opts.num_workers > 0,
                )

            train_loss_elbo = []
            earlystopper = EarlyStopper(
                mode="min", min_delta=0.1, patience=self._train_opts.early_stopper_patience, percentage=True
            )
            with self._train_opts.device:
                model.on_train_start(data)

            # needs to be after on_train_start
            optimizer = ClippedAdam(model.get_lr_func(self._train_opts.lr, lrd=lrd))
            svi = SVI(
                model=pyro.poutine.scale(model.model, scale=1.0 / self.n_samples_total),
                guide=pyro.poutine.scale(model.guide, scale=1.0 / self.n_samples_total),
                optim=optimizer,
                loss=TraceMeanField_ELBO(
                    retain_graph=True, num_particles=self._train_opts.n_particles, vectorize_particles=True
                ),
            )

        with tqdm(range(self._train_opts.max_epochs), unit="epoch", dynamic_ncols=True) as t:
            for i in t:
                with self._train_opts.device, torch.inference_mode():
                    model.on_train_epoch_start(i)

                epoch_loss = 0
                if singlebatch:
                    with self._train_opts.device:
                        epoch_loss += svi.step(**batchdata, **batch)
                else:
                    for batch in loader:
                        batch = collate((batch,), collate_fn_map=collate_fn_map)
                        with self._train_opts.device:
                            epoch_loss += svi.step(**batch.pop("data"), **batch)

                with self._train_opts.device, torch.inference_mode():
                    model.on_train_epoch_end(i)

                train_loss_elbo.append(epoch_loss)
                t.set_postfix({"Loss": epoch_loss}, refresh=False)

                if earlystopper.step(epoch_loss):
                    _logger.info(f"Training converged after {i} epochs.")
                    break

            if isinstance(t, tqdm_notebook):  # https://github.com/tqdm/tqdm/issues/1659
                t.container.children[1].bar_style = "success"

            with self._train_opts.device, torch.inference_mode():
                model.on_train_end(data, batch_size=self._train_opts.batch_size)

        self._train_loss_elbo = np.asarray(train_loss_elbo)
        self._model = model
        self._init_api()

        if self._train_opts.save_path is not False:
            if self._train_opts.save_path is None:
                self._train_opts.save_path = f"mofaflex_{time.strftime('%Y%m%d_%H%M%S')}.h5"
            else:
                self._train_opts.save_path = str(self._train_opts.save_path)
            _logger.info(f"Saving results to {self._train_opts.save_path}...")
            Path(self._train_opts.save_path).parent.mkdir(parents=True, exist_ok=True)
            self._save(self._train_opts.save_path)

    @_check_trained
    def get_r2(
        self, type: Literal["total", "byterm", "term"] | None = None, ordered: bool = False, term: str | None = None
    ) -> pd.DataFrame:
        """Get the fraction of explained variance for each view and group.

        Args:
            type: How fine-grained the fraction of explained variance should be split up.

                - `total`: Returns the total fraction of explained variance.
                - `byterm`: Returns the fraction of explained variance for each additive term.
                - `term`: Returns the fraction of explained variance for each component (e.g. factor) of the given term.

                Defaults to `term` if the model has only one additive term, `byterm` otherwise.
            ordered: Whether to sort the returned dataframes by explained variance (highest to lowest, per group and view).
                Has no effect for `type="total"`.
            term: The name of the additive term for `type="term"`. Can be `None` if the model has only one term.
        """
        if type is None:
            if self.n_terms == 1:
                type = "term"
            else:
                type = "byterm"

        if type == "term":
            if term is None:
                if self.n_terms > 1:
                    raise ValueError("Name of term required for 'type=term'.")
                else:
                    term = next(iter(self._wrapped_terms.keys()))

        return self._model.get_r2(type, ordered, term)

    @_check_trained
    def get_dispersion(self, moment: Literal["mean", "std"] = "mean") -> dict[str, pd.Series]:
        """Get the dispersion vectors for each view.

        Args:
            moment: Which moment of the posterior distribution to return.
        """
        return self._model.get_dispersion(self.feature_names, moment)

    @_check_trained
    def impute_data(
        self, data: MuData | Mapping[str, Mapping[str, AnnData]] | AnnData, missing_only=False
    ) -> dict[dict[str, AnnData]]:
        """Impute values in the training data using the trained factorization.

        The data will be transformed into a space compatible with model predictions. Usually that involves shifting and/or
        scaling, e.g. Gaussian data will be mean-centered and scaled to unit variance. This also implies that only dense
        matrices can be returned. Be aware that this can result in high memory consumption.

        Args:
            data: The data the model was trained on.
            missing_only: Only impute missing values in the data.

        Returns:
            Nested dictionary of AnnData objects with either fully imputed data or with only the missing values filled in.
        """
        data = self._make_dataset(data)
        data.reindex_features(self.feature_names)
        return self._model.impute(data, missing_only)

    def _save(self, path: str | Path):
        state = {
            "train_loss_elbo": self._train_loss_elbo,
            "group_names": self._group_names,
            "view_names": self._view_names,
            "feature_names": self._feature_names,
            "sample_names": self._sample_names,
            "metadata": self._metadata,
            "data_opts": self._data_opts.asdict(),
            "train_opts": self._train_opts.asdict(),
            "model": self._model.save(),
        }
        state["train_opts"]["device"] = str(state["train_opts"]["device"])

        save_model(state, path)

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "MOFAFLEX":
        """Load a saved MOFAFLEX model.

        Args:
            path: Path to the saved model file.
            map_location: Specify how to remap storage locations for PyTorch tensors. See the `torch.load`
                documentation for details.
        """
        state = load_model(path)

        if map_location is not None:
            state["train_opts"]["device"] = map_location

        model = cls.__new__(cls)
        model._train_loss_elbo = state["train_loss_elbo"]
        model._group_names = state["group_names"]
        model._view_names = state["view_names"]
        model._feature_names = state["feature_names"]
        model._sample_names = state["sample_names"]
        model._metadata = state["metadata"]
        model._data_opts = _DataOptions(**state["data_opts"])
        model._train_opts = _TrainingOptions(**state["train_opts"])

        with model._no_check_trained():
            model._model = MofaFlexModel.load(
                state["model"],
                map_location=map_location,
                sample_names=model.sample_names,
                feature_names=model.feature_names,
                n_samples=model.n_samples,
                n_features=model.n_features,
            )
        model._init_api()

        return model
