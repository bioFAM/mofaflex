import logging
import operator
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Any, get_args

import numpy as np
import pandas as pd
import pyro
import torch
from numpy.typing import NDArray
from pyro.nn import PyroModule, pyro_method
from scipy.sparse import issparse

from .api.likelihoods import Likelihood as APILikelihood
from .datasets import MofaFlexDataset, StackDataset
from .likelihoods import Likelihood, LikelihoodType
from .terms import Term
from .utils import PyroModuleDict

_logger = logging.getLogger(__name__)


class MofaFlexModel(PyroModule):
    """The MOFA-FLEX model.

    The model consists of multiple additive terms and a likelihood. Each additive term is responsible for handling its own
    parameters and state opaquely to the overall model.

    Args:
        terms: The additive terms.
        likelihoods: The likelhood for each view (if a mapping) or for all views otherwise.
    """

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
    def _group_names(self):
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
            if (dsets := term.get_datasets(data)) is not None and len(dsets)
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

    def get_lr_func(self, base_lr: float, **kwargs):
        modifiers = {}
        for term_name, term in self._terms.items():
            modifiers.update({f"_terms.{term_name}.{pname}": mod for pname, mod in term.learning_rate_multipliers})

        def lr_func(param_name):
            return dict(lr=base_lr * modifiers.get(param_name, 1), **kwargs)

        return lr_func

    def on_train_start(self, data: MofaFlexDataset):
        """Hook that is called immediately prior to training."""
        for term in self._terms.values():
            term.on_train_start(data)

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
            sample_names:
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
            cdata = data.preprocessor(view.X[sample_idx, :], slice(None), slice(None), group_name, view_name)[0]
            if issparse(cdata):
                cdata = cdata.toarray()

            alignment_idx = map_local_indices_to_global(  # noqa: F821
                slice(None), group_name, view_name, align_to="features"
            )
            try:
                r2_full = self._likelihoods[view_name].r2(
                    y_true=cdata,
                    y_pred=self.predict(group_name, view_name, sample_idx),
                    group_name=group_name,
                    alignment_idx=alignment_idx,
                )
                r2s_per_term = {}
                r2s_per_term_component = {}
                for term_name, term in self._terms.items():
                    r2s_per_term[term_name] = self._likelihoods[view_name].r2(
                        y_true=cdata,
                        y_pred=term.predict(group_name, view_name, sample_idx),
                        group_name=group_name,
                        alignment_idx=alignment_idx,
                    )

                    component_iter = term.prediction_components(group_name, view_name, sample_idx)
                    if component_iter is not None:
                        r2s_per_term_component[term_name] = {
                            component_name: self._likelihoods[view_name].r2(
                                y_true=cdata, y_pred=component, group_name=group_name, alignment_idx=alignment_idx
                            )
                            for component_name, component in component_iter
                        }
                return r2_full, r2s_per_term, r2s_per_term_component
            except NotImplementedError:
                _logger.warning(
                    f"R2 calculation for {self._model_opts.likelihoods[view_name]} likelihood has not yet been implemented. Skipping view {view_name} for group {group_name}."
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
                -components.groupby(["component"], sort=False)["R2"].mean().to_numpy()
            )

    def predict(self, group_name: str, view_name: str, subset_idx: NDArray[int] | slice = slice(None)):
        """Create a prediction for a given group and view.

        Args:
            group_name: The group.
            view_name: The view.
            subset_idx: The subset of samples to predict for.
        """
        return reduce(operator.add, (term.predict(group_name, view_name, subset_idx) for term in self._terms.values()))

    def save(self) -> dict[str, Any]:
        return {
            "terms": {name: term.save() for name, term in self._terms.items()},
            "likelihoods": {view_name: likelihood.save() for view_name, likelihood in self._likelihoods.items()},
        }

    @classmethod
    def load(self, state: dict[str, Any], n_samples: dict[str, int], n_features: dict[str, int], map_location=None):
        self._terms = PyroModuleDict(
            {name: Term.load(term, n_samples, n_features) for name, term in state["terms"].items()}
        )
        self._likelihoods = {
            view_name: Likelihood.load(likelihood, n_samples, n_features)
            for view_name, likelihood in state["likelihoods"].items()
        }
