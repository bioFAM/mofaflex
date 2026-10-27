from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from array_api_compat import array_namespace
from scipy.sparse import issparse
from scipy.stats import nbinom

from ..datasets import MofaFlexDataset
from ..settings import settings
from ..utils import Matrix, Vector, nanmean, nanmin
from .base import R2, Likelihood, LogLikelihoods
from .pyro import Likelihood as PyroLikelihood
from .pyro import NegativeBinomial as PyroNegativeBinomial


class NegativeBinomial(Likelihood):
    """Negative binomial likelhood for count data."""

    _priority = 5
    _state_attrs = ("_shift", "_sample_means", "_dispersion")

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool):
        super().__init__(view_name, data, nonnegative)
        sample_means = data.apply_to_view(
            view_name, lambda adata, group_name: nanmean(adata.layers[None], axis=1, keepdims=True)
        )
        statfun = nanmean if not nonnegative else nanmin
        self._shift = data.apply_to_view(
            view_name,
            lambda adata, group_name: align_local_array_to_global(  # noqa: F821
                statfun(adata.layers[None] / sample_means[group_name], axis=0),
                group_name,
                self._view_name,
                align_to="features",
            ),
        )
        self._sample_means = {
            group_name: data.align_local_array_to_global(gmeans, group_name, self._view_name, align_to="samples")
            for group_name, gmeans in sample_means.items()
        }
        self._dispersion = None

    def _get_pyro_likelihood(
        self,
        data: MofaFlexDataset,
        sample_dim: int,
        feature_dim: int,
        *,
        init_loc: float = np.e,
        init_scale: float = 0.1,
    ) -> PyroLikelihood:
        return PyroNegativeBinomial(
            self._view_name,
            sample_dim,
            feature_dim,
            data.n_samples,
            data.n_features[self._view_name],
            self._sample_means,
            shift=self._shift,
            init_loc=init_loc,
            init_scale=init_scale,
        )

    def on_train_end(self, *args, **kwargs):
        self._dispersion = self._pyro_likelihood.dispersion
        self._shift = self._impute_feature_summary_statistics(self._shift)
        for means in self._sample_means.values():
            means[np.isnan(means)] = np.nanmean(means)

    @classmethod
    def _validate(cls, adata: AnnData, view_name: str) -> str | None:
        if adata.layers.keys() != {None}:
            return f"The {cls.__name__} likelihood requires one layer, got {len(adata.layers)} for view {view_name}."
        data = adata.layers[None]
        if issparse(data):
            data = data.data
        xp = array_namespace(data)
        data = data[~xp.isnan(data)]
        if not xp.allclose(data, xp.round(data)) or data.min() < 0:
            return f"The {cls.__name__} likelihood in view {view_name} must be used with count (non-negative integer) data."
        return None

    def _r2_impl(
        self,
        y_true: Mapping[str | None, Matrix[np.number]],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> R2:
        y_true = y_true[None]
        dispersion = self._dispersion.mean[feature_idx]
        ss_res = np.nansum(self._dV_square(y_true, y_pred, dispersion, 1))

        truemean = self._shift[group_name][feature_idx] * self._sample_means[group_name][sample_idx]

        # use estimated dispersion to get R2=0 for y_pred=0
        ss_tot = np.nansum(self._dV_square(y_true, truemean, dispersion, 1))

        return R2(ss_res, ss_tot)

    def _logpmf(
        self,
        y_true: Mapping[str | None, Matrix[np.number]],
        mean: Matrix[np.floating],
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating]:
        y_true = y_true[None]
        dispersion = self._dispersion.mean[feature_idx]
        return nbinom.logpmf(y_true, p=1 / (1 + dispersion * mean + settings.eps), n=1 / dispersion)

    def _deviance_explained_impl(
        self,
        y_true: Mapping[str | None, Matrix[np.number]],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> LogLikelihoods:
        y_true = y_true[None]
        y_pred = self.transform_prediction(y_pred, group_name, sample_idx, feature_idx)

        null_mean = self._shift[group_name][feature_idx] * self._sample_means[group_name][sample_idx]

        # use the estimated dispersion for all three models, otherwise the log-likelihoods are not comparable
        loglik_saturated = np.nansum(self._logpmf(y_true, y_true, feature_idx))
        loglik_null = np.nansum(self._logpmf(y_true, null_mean, feature_idx))
        loglik_model = np.nansum(self._logpmf(y_true, y_pred, feature_idx))

        return LogLikelihoods(loglik_saturated, loglik_null, loglik_model)

    def transform_prediction(
        self,
        prediction: Matrix[np.floating] | Vector[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating] | Vector[np.floating]:
        prediction = prediction + self._shift[group_name][feature_idx]
        prediction = np.maximum(0, prediction)  # ReLU
        sample_means = self._sample_means[group_name][sample_idx]
        if prediction.ndim == 1 and prediction.size == sample_idx.size == feature_idx.size:
            sample_means = sample_means.squeeze(-1)
        prediction *= sample_means
        return prediction

    def transform_data(
        self,
        data: Mapping[str | None, Matrix[np.number]],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating]:
        data = data[None] / self._sample_means[group_name][sample_idx]
        data -= self._shift[group_name][feature_idx]
        return data

    @Likelihood._api
    def get_dispersion(self, moment: Literal["mean", "std"] = "mean") -> pd.Series:
        """Get the dispersion vectors for each view.

        Args:
            moment: Which moment of the posterior distribution to return.
        """
        return pd.Series(getattr(self._dispersion, moment), index=self._feature_names)
