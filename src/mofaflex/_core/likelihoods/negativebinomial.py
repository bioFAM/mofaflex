import numpy as np
from numpy.typing import NDArray

from .. import utils
from ..datasets import MofaFlexDataset
from .base import R2, Likelihood
from .pyro import PyroLikelihood, PyroNegativeBinomial


class NegativeBinomial(Likelihood):
    _priority = 5
    _state_attrs = ("_shift", "_sample_means", "_dispersion")

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool):
        super().__init__(view_name, data, nonnegative)
        self._shift = data.apply_to_view(view_name, lambda adata, group_name: utils.nanmean(adata.X, axis=0))
        self._sample_means = data.apply_to_view(view_name, lambda adata, group_name: utils.nanmean(adata.X, axis=1))
        self._dispersion = None

    def _get_pyro_likelihood(
        self,
        data: MofaFlexDataset,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        *,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
    ) -> PyroLikelihood:
        sample_means = {
            group_name: data.align_local_array_to_global(gmeans, group_name, self._view_name, align_to="samples")
            for group_name, gmeans in self._sample_means.items()
        }
        return PyroNegativeBinomial(
            view_name,
            sample_dim,
            feature_dim,
            data.n_samples,
            data.n_features[self._view_name],
            self._shift,
            sample_means,
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

    def _r2_impl(self, y_true: NDArray, y_pred: NDArray[np.floating], group_name: str):
        ss_res = np.nansum(self._dV_square(y_true, y_pred, self._dispersion.mean[group_name], 1))

        truemean = np.nanmean(y_true)
        nu2 = (np.nanvar(y_true) - truemean) / truemean**2  # method of moments estimator
        ss_tot = np.nansum(self._dV_square(y_true, truemean, nu2, 1))

        return R2(ss_res, ss_tot)

    def transform_prediction(self, prediction: NDArray[np.floating], group_name: str):
        prediction = prediction + self._shift[group_name]
        prediction = np.maximum(0, prediction)  # ReLU
        prediction *= self._sample_means[group_name][..., None]
        return prediction
