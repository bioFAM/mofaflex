import numpy as np
from numpy.typing import NDArray

from .. import utils
from ..datasets import MofaFlexDataset
from .base import R2, Likelihood
from .pyro import Likelihood as PyroLikelihood
from .pyro import NegativeBinomial as PyroNegativeBinomial


class NegativeBinomial(Likelihood):
    """Negative binomial likelhood for count data."""

    _priority = 5
    _state_attrs = ("_shift", "_sample_means", "_dispersion")

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool):
        super().__init__(view_name, data, nonnegative)
        sample_means = data.apply_to_view(
            view_name, lambda adata, group_name: utils.nanmean(adata.X, axis=1, keepdims=True)
        )
        statfun = utils.nanmean if not nonnegative else utils.nanmin
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

    @classmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        return xp.allclose(data, xp.round(data)) and data.min() >= 0

    @classmethod
    def _format_validate_exception(cls, view_name: str) -> str:
        return f"NegativeBinomial likelihood in view {view_name} must be used with count (non-negative integer) data."

    def _r2_impl(
        self,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ):
        ss_res = np.nansum(self._dV_square(y_true, y_pred, self._dispersion.mean[feature_idx], 1))

        truemean = self._shift[group_name][feature_idx]
        nu2 = (np.nanvar(y_true, axis=0) - truemean) / truemean**2  # method of moments estimator
        ss_tot = np.nansum(self._dV_square(y_true, truemean, nu2, 1))

        return R2(ss_res, ss_tot)

    def transform_prediction(
        self,
        prediction: NDArray[np.floating],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ):
        prediction = prediction + self._shift[group_name][feature_idx]
        prediction = np.maximum(0, prediction)  # ReLU
        prediction *= self._sample_means[group_name][sample_idx]
        return prediction

    def transform_data(
        self,
        data: NDArray[np.number],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ):
        data = data / self._sample_means[group_name][sample_idx]
        data -= self._shift[group_name][feature_idx]
        return data

    @property
    def dispersion(self):
        return self._dispersion
