from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from .. import utils
from ..datasets import MofaFlexDataset
from ..pyro.likelihoods import PyroLikelihood, PyroNegativeBinomial
from .base import R2, Likelihood


class NegativeBinomial(Likelihood):
    _priority = 5

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool):
        super().__init__(view_name, data, nonnegative)
        self._shift = data.apply_to_view(view_name, lambda adata, group_name: utils.nanmin(adata.X, axis=0))
        self._sample_means = data.apply_to_view(
            view_name,
            lambda adata, group_name: align_local_array_to_global(  # noqa: F821
                utils.nanmean(adata.X, axis=1), group_name, view_name, align_to="samples"
            ),
        )

    def pyro_likelihood(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        nsamples: Mapping[str, int],
        nfeatures: int,
        *,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        **kwargs,
    ) -> PyroLikelihood:
        return PyroNegativeBinomial(
            view_name,
            sample_dim,
            feature_dim,
            nsamples,
            nfeatures,
            self._shift,
            self._sample_means,
            init_loc=init_loc,
            init_scale=init_scale,
        )

    @classmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        return xp.allclose(data, xp.round(data)) and data.min() >= 0

    @classmethod
    def _format_validate_exception(cls, view_name: str) -> str:
        return f"NegativeBinomial likelihood in view {view_name} must be used with count (non-negative integer) data."

    @classmethod
    def _r2_impl(
        cls,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ):
        ss_res = np.nansum(cls._dV_square(y_true, y_pred, dispersions, 1))

        truemean = np.nanmean(y_true)
        nu2 = (np.nanvar(y_true) - truemean) / truemean**2  # method of moments estimator
        ss_tot = np.nansum(cls._dV_square(y_true, truemean, nu2, 1))

        return R2(ss_res, ss_tot)

    def transform_prediction(self, prediction: NDArray[np.floating], group_name: str):
        prediction = np.maximum(0, prediction)  # ReLU
        prediction *= self._sample_means[..., None] + self._shift[group_name]
        return prediction
