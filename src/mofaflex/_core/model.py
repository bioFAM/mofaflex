import operator
from collections.abc import Mapping
from functools import reduce

import pyro
import torch
from numpy.typing import NDArray
from pyro.nn import PyroModule, pyro_method

from .datasets import MofaFlexDataset, StackDataset
from .likelihoods import Likelihood
from .pyro.utils import PyroModuleDict
from .terms import Term
from .utils import MeanStd


class MofaFlexModel(PyroModule):
    _sample_plate_dim = -2
    _feature_plate_dim = -1

    def __init__(
        self,
        n_samples: Mapping[str, int],
        n_features: Mapping[str, int],
        terms: Mapping[str, Term],
        likelihoods: Mapping[str, Likelihood],
        sample_means: Mapping[str, Mapping[str, NDArray]] = None,
        feature_means: Mapping[str, Mapping[str, NDArray]] = None,
    ):
        super().__init__()
        self._n_samples = n_samples
        self._n_features = n_features

        self._terms = PyroModuleDict(terms)
        self._likelihoods = PyroModuleDict(
            {
                view_name: likelihood.pyro_likelihood(
                    view_name=view_name,
                    sample_dim=self._sample_plate_dim,
                    feature_dim=self._feature_plate_dim,
                    sample_means=sample_means,
                    feature_means=feature_means,
                )
                for view_name, likelihood in likelihoods.items()
            }
        )

        self._scale_elbo = True
        n_views = len(self._view_names)
        self._view_scales = dict.fromkeys(self._view_names, 1.0)
        if self._scale_elbo and n_views > 1:
            for view_name, view_n_features in n_features.items():
                self._view_scales[view_name] = (n_views / (n_views - 1)) * (
                    1.0 - view_n_features / sum(n_features.values())
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

                    self._likelihoods[view_name].model(
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
        return {
            termname: StackDataset(**dsets)
            for termname, term in self._terms.items()
            if (dsets := term.get_datasets(data)) is not None and len(dsets)
        }

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
                self._likelihoods[view_name].guide(group_name, sample_plates[group_name], feature_plates[view_name])

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

    def on_train_epoch_start(self, epoch: int):
        """Hook that is called at the beginning of each epoch.

        Args:
            epoch: The current epoch.
        """
        for term in self._terms.values():
            term.on_train_epoch_start(epoch)

    def on_train_epoch_end(self, epoch: int):
        """Hook that is called at the end of each epoch.

        Args:
            epoch: The current epoch.
        """
        for term in self._terms.values():
            term.on_train_epoch_end(epoch)

    def on_train_end(self, data: MofaFlexDataset, batch_size: int):
        """Hook that is called at the end of training.

        Args:
            data: The dataset used during training.
            sample_names:
            batch_size: The batch size used during training.
        """
        for term in self._terms.values():
            term.on_train_end(data, batch_size)

    @torch.inference_mode()
    def get_dispersion(self):
        """Get all dispersion vectors, dispersion_x."""
        dispersion = MeanStd({}, {})
        for view_name, likelihood in self._likelihoods.items():
            try:
                disp = likelihood.dispersion
            except AttributeError:
                continue
            dispersion.mean[view_name] = disp.mean
            dispersion.std[view_name] = disp.std

        return dispersion

    def predict(self, group_name: str, view_name: str, subset_idx: NDArray[int] | None = None):
        return reduce(operator.add, (term.predict(group_name, view_name, subset_idx) for term in self._terms))
