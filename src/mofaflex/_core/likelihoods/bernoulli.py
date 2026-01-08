import numpy as np
from numpy.typing import NDArray
from scipy.special import expit, logit

from .. import utils
from ..datasets import MofaFlexDataset
from .base import R2, Likelihood
from .pyro import PyroBernoulli, PyroLikelihood


class Bernoulli(Likelihood):
    """Bernoulli likelihood for binary data."""

    _priority = 10
    _state_attrs = ("_shift",)

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool):
        super().__init__(view_name, data, nonnegative)
        self._shift = data.apply_to_view(view_name, lambda adata, group_name: logit(utils.nanmean(adata.X, axis=0)))

    def _get_pyro_likelihood(self, data: MofaFlexDataset, sample_dim: int, feature_dim: int) -> PyroLikelihood:
        return PyroBernoulli(
            self._view_name,
            sample_dim,
            feature_dim,
            data.n_samples,
            data.n_features[self._view_name],
            shift=self._shift,
        )

    @classmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        return xp.all(xp.isclose(data, 0) | xp.isclose(data, 1))  # TODO: set correct atol value

    @classmethod
    def _format_validate_exception(cls, view_name: str) -> str:
        return f"Bernoulli likelihood in view {view_name} must be used with binary data."

    @classmethod
    def _r2_impl(
        cls,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> R2:
        ss_res = np.nansum(cls._dV_square(y_true, y_pred, -1, 1))
        ss_tot = np.nansum(cls._dV_square(y_true, np.nanmean(y_true), -1, 1))
        return R2(ss_res, ss_tot)

    def transform_prediction(self, prediction: NDArray[np.floating], group_name: str):
        return expit(prediction + self._shift[group_name])
