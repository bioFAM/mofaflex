from typing import Literal

import numpy as np
import pandas as pd

from ..datasets import MofaFlexDataset
from ..utils import Matrix, Vector, nanmean, nanmin
from .base import R2, Likelihood
from .pyro import Likelihood as PyroLikelihood
from .pyro import NegativeBinomial as PyroNegativeBinomial


class NegativeBinomial(Likelihood):
    """Negative binomial likelhood for count data."""

    _priority = 5
    _state_attrs = ("_shift", "_sample_means", "_dispersion")

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool):
        super().__init__(view_name, data, nonnegative)
        sample_means = data.apply_to_view(view_name, lambda adata, group_name: nanmean(adata.X, axis=1, keepdims=True))
        statfun = nanmean if not nonnegative else nanmin
        self._shift = data.apply_to_view(
            view_name,
            lambda adata, group_name: align_local_array_to_global(  # noqa: F821
                statfun(adata.X / sample_means[group_name], axis=0), group_name, self._view_name, align_to="features"
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
    def _validate(cls, data: Matrix, xp) -> bool:
        return xp.allclose(data, xp.round(data)) and data.min() >= 0

    @classmethod
    def _format_validate_exception(cls, view_name: str) -> str:
        return f"NegativeBinomial likelihood in view {view_name} must be used with count (non-negative integer) data."

    def _r2_impl(
        self,
        y_true: Matrix,
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> R2:
        ss_res = np.nansum(self._dV_square(y_true, y_pred, self._dispersion.mean[feature_idx], 1))

        truemean = self._shift[group_name][feature_idx]
        nu2 = (np.nanvar(y_true, axis=0, mean=truemean) - truemean) / truemean**2  # method of moments estimator
        ss_tot = np.nansum(self._dV_square(y_true, truemean, nu2, 1))

        return R2(ss_res, ss_tot)

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
        data: Matrix[np.number],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating]:
        data = data / self._sample_means[group_name][sample_idx]
        data -= self._shift[group_name][feature_idx]
        return data

    @Likelihood._api
    def get_dispersion(self, moment: Literal["mean", "std"] = "mean") -> pd.Series:
        """Get the dispersion vectors for each view.

        Args:
            moment: Which moment of the posterior distribution to return.
        """
        return pd.Series(getattr(self._dispersion, moment), index=self._feature_names)
