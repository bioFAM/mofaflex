import logging
import operator
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from functools import reduce
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import pyro
import torch
from anndata import AnnData
from numpy.typing import NDArray
from pyro.nn import PyroModule, pyro_method
from scipy.sparse import issparse

from .api.likelihoods import Likelihood as APILikelihood
from .datasets import MofaFlexDataset, StackDataset
from .likelihoods import Likelihood, LikelihoodType
from .settings import settings
from .terms import Term
from .utils import PyroModuleDict, SaveStateMixin, wherenan

_logger = logging.getLogger(__name__)


class MofaFlexModel(SaveStateMixin, PyroModule):
    """The MOFA-FLEX model.

    The model consists of multiple additive terms and a likelihood. Each additive term is responsible for handling its own
    parameters and state opaquely to the overall model.

    Args:
        terms: The additive terms.
        likelihoods: The likelhood for each view (if a mapping) or for all views otherwise.
    """

    _state_attrs = ("_r2_full", "_r2_terms", "_r2_term_components", "_term_order")
    _sample_plate_dim = -2
    _feature_plate_dim = -1

    def __init__(
        self,
        terms: Mapping[str, Term],
        likelihoods: Mapping[str | Sequence[str], LikelihoodType | APILikelihood]
        | LikelihoodType
        | APILikelihood
        | None,
    ):
        super().__init__()

        self._terms = PyroModuleDict(terms)
        self._likelihoods = likelihoods

    def _init(self, data: MofaFlexDataset):
        self._n_samples = data.n_samples
        self._n_features = data.n_features
        self._scale_elbo = True
        n_views = len(self._view_names)
        self._view_scales = dict.fromkeys(self._view_names, 1.0)
        if self._scale_elbo and n_views > 1:
            for view_name, view_n_features in data.n_features.items():
                self._view_scales[view_name] = (n_views / (n_views - 1)) * (
                    1.0 - view_n_features / data.n_features_total
                )

        nonnegative_views = set()
        nonnegative_terms = [term.nonnegative for term in self._terms.values()]
        for view_name in data.view_names:
            if all(all(group[view_name] for group in term.values()) for term in nonnegative_terms):
                nonnegative_views.add(view_name)

        if (
            not isinstance(self._likelihoods, dict | str | APILikelihood | None)
            or isinstance(self._likelihoods, str)
            and self._likelihoods not in get_args(LikelihoodType)
            or isinstance(self._likelihoods, dict)
            and not all(
                isinstance(val, APILikelihood) or val in get_args(LikelihoodType) for val in self._likelihoods.values()
            )
        ):
            raise ValueError(
                "Likelihoods must be a dictionary or a string containing a valid likelihood name or a Likelihood instance."
            )

        if self._likelihoods is None:
            self._likelihoods = data.apply(Likelihood.infer, by_group=False)
            msg = []
            for view_name, likelihood in self._likelihoods.items():
                msg.append(f"{view_name}: {likelihood.__name__}")
                self._likelihoods[view_name] = likelihood(view_name, data, view_name in nonnegative_views)
            _logger.info("No likelihoods provided. Using inferred likelihoods: " + "; ".join(msg))
        else:
            if isinstance(self._likelihoods, str | APILikelihood):
                self._likelihoods = dict.fromkeys(data.view_names, self._likelihoods)

            likelihoods = {}
            for views, likelihood in self._likelihoods.items():
                if isinstance(views, str):
                    views = (views,)
                for view in views:
                    likelihoods[view] = (
                        Likelihood(likelihood, view, data, view_name in nonnegative_views)
                        if isinstance(likelihood, str)
                        else likelihood(view, data, view_name in nonnegative_views)
                    )
            self._likelihoods = likelihoods
            data.apply(
                lambda *args, likelihood, **kwargs: likelihood.validate(*args, **kwargs),
                view_kwargs={"likelihood": self._likelihoods},
                by_group=False,
            )

    @property
    def terms(self) -> Mapping[str, Term]:
        return self._terms

    @property
    def _group_names(self) -> Sequence[str]:
        return self._n_samples.keys()

    @property
    def _view_names(self):
        return self._n_features.keys()

    def _get_plates(self, subsample=None):
        sample_plates = {}

        for group_name in self._group_names:
            sample_plates[group_name] = pyro.plate(
                f"plate_samples_{group_name}",
                self._n_samples[group_name],
                dim=self._sample_plate_dim,
                subsample=subsample[group_name],
            )

        feature_plates = {}
        for view_name in self._view_names:
            feature_plates[view_name] = pyro.plate(
                f"plate_features_{view_name}",
                self._n_features[view_name],
                subsample=torch.arange(  # workaround for https://github.com/pyro-ppl/pyro/pull/3405
                    self._n_features[view_name]
                ),
                dim=self._feature_plate_dim,
            )

        return sample_plates, feature_plates

    @pyro_method
    def model(self, data, sample_idx, nonmissing_samples, nonmissing_features, **kwargs):
        sample_plates, feature_plates = self._get_plates(subsample=sample_idx)

        predictions = [
            term.model(
                sample_plates, feature_plates, nonmissing_samples, nonmissing_features, **kwargs.get(termname, {})
            )
            for termname, term in self._terms.items()
        ]

        for group_name, group in data.items():
            gnonmissing_samples = nonmissing_samples[group_name]
            gnonmissing_features = nonmissing_features[group_name]
            for view_name, view in group.items():
                if view.numel() == 0:  # can occur in the last batch of an epoch if the batch is small
                    continue
                prediction = None
                for term in predictions:
                    try:
                        term_prediction = term[group_name][view_name]
                    except KeyError:
                        continue
                    if prediction is None:
                        prediction = term_prediction
                    else:
                        prediction += term_prediction
                if prediction is not None:
                    vnonmissing_samples = gnonmissing_samples[view_name]
                    vnonmissing_features = gnonmissing_features[view_name]

                    self._pyro_likelihoods[view_name].model(
                        data=view,
                        estimate=prediction,
                        group_name=group_name,
                        scale=self._view_scales[view_name],
                        sample_plate=sample_plates[group_name],
                        feature_plate=feature_plates[view_name],
                        nonmissing_samples=vnonmissing_samples,
                        nonmissing_features=vnonmissing_features,
                    )

    def get_datasets(self, data: MofaFlexDataset) -> dict[str, StackDataset]:
        """Hook that is called prior to training.

        If a prior requires any additional covariates during training, it should return a dict of datasets. The keys of
        the dict will be used as argument names for the `model` and `guide` methods of the Pyro prior.

        Args:
            data: The dataset.
        """
        dsets = {
            termname: StackDataset(**dsets)
            for termname, term in self._terms.items()
            if (dsets := term.get_datasets(data, self._sample_plate_dim, self._feature_plate_dim)) is not None
            and len(dsets)
        }

        self._init(data)
        return dsets

    @pyro_method
    def guide(self, data, sample_idx, nonmissing_samples, nonmissing_features, **kwargs):
        sample_plates, feature_plates = self._get_plates(subsample=sample_idx)
        for termname, term in self._terms.items():
            term.guide(
                sample_plates, feature_plates, nonmissing_samples, nonmissing_features, **kwargs.get(termname, {})
            )

        for group_name, group in data.items():
            for view_name, view_obs in group.items():
                if view_obs.numel() == 0:
                    continue
                self._pyro_likelihoods[view_name].guide(
                    group_name, sample_plates[group_name], feature_plates[view_name]
                )

    def get_lr_func(self, base_lr: float, **kwargs) -> Callable[[str], Mapping[str, Any]]:
        """Get a learning rate function that can be passed to a Pyro optimizer.

        This is useful if some parameters need a different learning rate than the rest.

        Args:
            base_lr: The base learning rate.
            **kwargs: Additional arguments to the optimizer.
        """
        modifiers = {}
        for term_name, term in self._terms.items():
            modifiers.update({f"_terms.{term_name}.{pname}": mod for pname, mod in term.learning_rate_multipliers})

        def lr_func(param_name):
            return dict(lr=base_lr * modifiers.get(param_name, 1), **kwargs)

        return lr_func

    def on_train_start(self, data: MofaFlexDataset):
        """Hook that is called immediately prior to training."""
        for term in self._terms.values():
            term.on_train_start(data, self._sample_plate_dim, self._feature_plate_dim)

        self._pyro_likelihoods = PyroModuleDict(
            {
                view_name: likelihood.get_pyro_likelihood(
                    data, sample_dim=self._sample_plate_dim, feature_dim=self._feature_plate_dim
                )
                for view_name, likelihood in self._likelihoods.items()
            }
        )

        for likelihood in self._likelihoods.values():
            likelihood.on_train_start()

    def on_train_epoch_start(self, epoch: int):
        """Hook that is called at the beginning of each epoch.

        Args:
            epoch: The current epoch.
        """
        for term in self._terms.values():
            term.on_train_epoch_start(epoch)
        for likelihood in self._likelihoods.values():
            likelihood.on_train_epoch_start(epoch)

    def on_train_epoch_end(self, epoch: int):
        """Hook that is called at the end of each epoch.

        Args:
            epoch: The current epoch.
        """
        for term in self._terms.values():
            term.on_train_epoch_end(epoch)
        for likelihood in self._likelihoods.values():
            likelihood.on_train_epoch_end(epoch)

    def on_train_end(self, data: MofaFlexDataset, batch_size: int):
        """Hook that is called at the end of training.

        Args:
            data: The dataset used during training.
            batch_size: The batch size used during training.
        """
        for term in self._terms.values():
            term.on_train_end(data, batch_size)
        for likelihood in self._likelihoods.values():
            likelihood.on_train_end(data, batch_size)

        subsample = 1000  # TODO: or use the batch size

        def r2_wrapper(view, group_name, view_name):
            if subsample is not None and subsample > 0 and subsample < view.n_obs:
                sample_idx = np.random.choice(view.n_obs, subsample, replace=False)
            else:
                sample_idx = slice(None)
            mapped_sample_idx = map_local_indices_to_global(sample_idx, group_name, view_name, align_to="samples")  # noqa: F821
            cdata = data.preprocessor(view.X[sample_idx, :], slice(None), slice(None), group_name, view_name)[0]
            if issparse(cdata):
                cdata = cdata.toarray()

            mapped_feature_idx = map_local_indices_to_global(  # noqa: F821
                slice(None), group_name, view_name, align_to="features"
            )
            try:
                r2_full = self._likelihoods[view_name].r2(
                    y_true=cdata,
                    y_pred=self.predict(group_name, view_name, mapped_sample_idx, mapped_feature_idx),
                    group_name=group_name,
                    sample_idx=mapped_sample_idx,
                    feature_idx=mapped_feature_idx,
                )
                r2s_per_term = {}
                r2s_per_term_component = {}
                for term_name, term in self._terms.items():
                    r2s_per_term[term_name] = self._likelihoods[view_name].r2(
                        y_true=cdata,
                        y_pred=term.predict(group_name, view_name, mapped_sample_idx, mapped_feature_idx),
                        group_name=group_name,
                        sample_idx=mapped_sample_idx,
                        feature_idx=mapped_feature_idx,
                    )

                    component_iter = term.prediction_components(
                        group_name, view_name, mapped_sample_idx, mapped_feature_idx
                    )
                    if component_iter is not None:
                        r2s_per_term_component[term_name] = {
                            component_name: self._likelihoods[view_name].r2(
                                y_true=cdata,
                                y_pred=component,
                                group_name=group_name,
                                sample_idx=mapped_sample_idx,
                                feature_idx=mapped_feature_idx,
                            )
                            for component_name, component in component_iter
                        }
                if r2_full < settings.eps:
                    _logger.warning(
                        f"R2 for view {view_name} is 0. Adjust model parameters and/or increase the number of training epochs."
                    )
                return r2_full, r2s_per_term, r2s_per_term_component
            except NotImplementedError:
                _logger.warning(
                    f"R2 calculation for {self._likelihoods[view_name]} likelihood has not yet been implemented. Skipping view {view_name} for group {group_name}."
                )

        r2s = data.apply(r2_wrapper)

        df_full, df_terms, dfs_term_components = {}, {}, defaultdict(dict)
        for group_name, group_r2s in r2s.items():
            gfull_df = {}
            term_df = {}
            components_dfs = defaultdict(dict)
            for view_name, (r2_full, r2s_per_term, r2s_per_term_component) in group_r2s.items():
                gfull_df[view_name] = r2_full
                term_df[view_name] = pd.Series(r2s_per_term, name="R2")
                for term_name, term_components in r2s_per_term_component.items():
                    components_dfs[term_name][view_name] = pd.DataFrame(
                        {"component": term_components.keys(), "R2": term_components.values()}
                    )
            df_full[group_name] = pd.Series(gfull_df, name="R2")
            df_terms[group_name] = pd.concat(term_df, axis=0)
            for term_name, term_dfs in components_dfs.items():
                dfs_term_components[term_name][group_name] = (
                    pd.concat(term_dfs, axis=0).droplevel(1).reset_index(names="view")
                )
        self._r2_full = pd.concat(df_full, axis=0, names=("group", "view")).reset_index()
        self._r2_terms = pd.concat(df_terms, axis=0, names=("group", "view", "term")).reset_index()
        self._r2_term_components = {
            term_name: pd.concat(term_df, axis=0).droplevel(1).reset_index(names="group")
            for term_name, term_df in dfs_term_components.items()
        }

        for term_name, components in self._r2_term_components.items():
            self._terms[term_name].component_order = np.argsort(
                -components.groupby("component", sort=False)["R2"].mean().to_numpy()
            )

        self._term_order = np.argsort(-self._r2_terms.groupby("term", sort=False)["R2"].mean().to_numpy())

    def get_r2(
        self, type: Literal["total", "byterm", "term"] = "byterm", ordered: bool = False, term: str | None = None
    ) -> pd.DataFrame:
        """Get the fraction of explained variance for each view and group.

        Args:
            type: How fine-grained the fraction of explained variance should be split up.

                - `total`: Returns the total fraction of explained variance.
                - `byterm`: Returns the fraction of explained variance for each additive term.
                - `term`: Returns the fraction of explained variance for each component (e.g. factor) of the given term.
            ordered: Whether to sort the returned dataframe by explained variance (highest to lowest, per group and view).
                Has no effect for `type="total"`.
            term: The name of the additive term if `type="term"`.
        """
        if type == "term":
            if term is None:
                raise ValueError("Name of term required for 'type=term'.")
            ret = self._r2_term_components[term]
            if ordered:
                ret = (
                    ret.groupby(["group", "view"], sort=False, as_index=False, group_keys=False)
                    .apply(lambda df: df.iloc[self._terms[term].component_order, :])
                    .reset_index(drop=True)
                )
            return ret
        elif type == "total":
            return self._r2_full
        elif type == "byterm":
            return (
                self._r2_terms
                if not ordered
                else self._r2_terms.groupby(["group", "view"], sort=False, as_index=False, group_keys=False)
                .apply(lambda df: df.iloc[self._term_order, :])
                .reset_index(drop=True)
            )
        else:
            raise ValueError(f"Unknown type argument '{type}'")

    def predict(
        self,
        group_name: str,
        view_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ):
        """Create a prediction for a given group and view.

        Args:
            group_name: The group.
            view_name: The view.
            sample_idx: The subset of samples to predict for.
            feature_idx: The subset of features to predict for.
        """
        return reduce(
            operator.add,
            (term.predict(group_name, view_name, sample_idx, feature_idx) for term in self._terms.values()),
        )

    def get_dispersion(
        self, feature_names: Mapping[str, NDArray[str]], moment: Literal["mean", "std"] = "mean"
    ) -> dict[str, pd.Series]:
        """Get the dispersion vectors for each view.

        Args:
            feature_names: Feature names for each view
            moment: Which moment of the posterior distribution to return.
        """
        return {
            view_name: pd.Series(getattr(dispersion, moment), index=feature_names[view_name])
            for view_name, likelihood in self._likelihoods.items()
            if (dispersion := likelihood.dispersion) is not None
        }

    def _impute(
        self, data: AnnData, group_name, view_name, sample_names, feature_names, likelihood, missingonly, preprocessor
    ):
        havemissing = data.n_obs < self._n_samples[group_name] or data.n_vars < self._n_features[view_name]
        if issparse(data.X):
            have_missing_cells = np.isnan(data.X.data).sum() > 0
        else:
            have_missing_cells = np.isnan(data.X).sum() > 0
        havemissing |= have_missing_cells

        if missingonly and not havemissing:
            return data

        if not missingonly:
            imputation = self.predict(group_name, view_name)
        else:
            missing_obs = align_local_array_to_global(  # noqa F821
                np.broadcast_to(False, (data.n_obs,)), group_name, view_name, fill_value=True, align_to="samples"
            )
            missing_var = align_local_array_to_global(  # noqa F821
                np.broadcast_to(False, (data.n_vars)), group_name, view_name, fill_value=True, align_to="features"
            )

            preprocessed = preprocessor(data.X, slice(None), slice(None), group_name, view_name)[0]
            if issparse(preprocessed):
                preprocessed = preprocessed.toarray()

            obsidx = map_local_indices_to_global(np.arange(data.n_obs), group_name, view_name, align_to="samples")  # noqa: F821
            varidx = map_local_indices_to_global(np.arange(data.n_vars), group_name, view_name, align_to="features")  # noqa: F821
            imputation = np.empty((sample_names.size, feature_names.size), dtype=data.X.dtype)
            imputation[np.ix_(obsidx, varidx)] = likelihood.transform_data(preprocessed, group_name, obsidx, varidx)

            imputation[missing_obs, :] = self.predict(group_name, view_name, sample_idx=missing_obs)
            imputation[:, missing_var] = self.predict(group_name, view_name, feature_idx=missing_var)

            if have_missing_cells:
                nanobs, nanvar = wherenan(data.X)
                nanobs, nanvar = np.atleast_1d(obsidx[nanobs]), np.atleast_1d(varidx[nanvar])
                for nobs, nvar in zip(nanobs, nanvar, strict=True):
                    imputation[nobs, nvar] = self.predict(
                        group_name, view_name, np.atleast_1d(nobs), np.atleast_1d(nvar)
                    ).squeeze()

        return AnnData(X=imputation, obs=pd.DataFrame(index=sample_names), var=pd.DataFrame(index=feature_names))

    def impute(self, data: MofaFlexDataset, missing_only=False) -> dict[dict[str, AnnData]]:
        """Impute values in the training data using the trained factorization.

        Args:
            data: The data the model was trained on.
            missing_only: Only impute missing values in the data.

        Returns:
            Nested dictionary of AnnData objects with either fully imputed data or with only the missing values filled in.
            In both cases, the returned data will be preprocessed. In the case of Gaussian distributed data, that involves
            centering and scaling.
        """
        return data.apply(
            self._impute,
            view_kwargs={"feature_names": data.feature_names, "likelihood": self._likelihoods},
            group_kwargs={"sample_names": data.sample_names},
            missingonly=missing_only,
            preprocessor=data.preprocessor,
        )

    def _save(self) -> dict[str, Any]:
        return {
            "terms": {name: term.save() for name, term in self._terms.items()},
            "likelihoods": {view_name: likelihood.save() for view_name, likelihood in self._likelihoods.items()},
        }

    def _load(
        self,
        state: dict[str, Any],
        sample_names: Mapping[str, NDArray[str]],
        feature_names: Mapping[str, NDArray[str]],
        n_samples: Mapping[str, int],
        n_features: Mapping[str, int],
        map_location=None,
        **kwargs,
    ):
        self._terms = PyroModuleDict(
            {
                name: Term.load(
                    term,
                    sample_names=sample_names,
                    feature_names=feature_names,
                    n_samples=n_samples,
                    n_features=n_features,
                    map_location=map_location,
                    **kwargs,
                )
                for name, term in state["terms"].items()
            }
        )
        self._likelihoods = {
            view_name: Likelihood.load(likelihood, map_location=map_location)
            for view_name, likelihood in state["likelihoods"].items()
        }
