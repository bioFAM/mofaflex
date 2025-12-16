from collections.abc import Mapping

import numpy as np
from anndata import AnnData
from array_api_compat import array_namespace
from numpy.typing import NDArray

from .. import utils
from ..datasets import MofaFlexDataset
from ..pyro.likelihoods import PyroLikelihood, PyroNormal
from .base import R2, Likelihood


class Normal(Likelihood):
    _priority = 0

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool, scale_per_group: bool = True):
        super().__init__(view_name, data, nonnegative)
        self._scale_per_group = scale_per_group
        statfun = utils.nanmean if not nonnegative else utils.nanmin
        self._shift = data.apply_to_view(view_name, lambda adata, group_name: statfun(adata.X, axis=0))

        if scale_per_group:
            self._scale = data.apply_to_view(view_name, self._calc_scale_grouped)
        else:
            self._scale = data.apply(self._calc_scale_ungrouped, by_group=False, filter_views=view_name)

    def _calc_scale_ungrouped(self, adata: AnnData, group: NDArray[object], view_name: str, groups: list[str]):
        if adata.n_obs <= 1:
            return 1.0

        arr = adata.X.copy()
        for group_name in groups:
            arr[group == group_name] -= align_local_array_to_global(  # noqa F821
                self._shift[group_name], group_name, view_name, align_to="features", axis=0
            )
        return np.sqrt(utils.nanvar(arr, axis=None))

    def _calc_scale_grouped(self, adata: AnnData, group_name: str):
        arr = adata.X - np.broadcast_to(
            self._shift[group_name], adata.X.shape
        )  # need to manually broadcast to force sparse to autoconvert to dense instead of raising
        if isinstance(arr, np.matrix):
            arr = np.asarray(arr)
        arr = utils.nanvar(arr, axis=None)
        xp = array_namespace(arr)
        return xp.sqrt(arr)

    def pyro_likelihood(
        self,
        sample_dim: int,
        feature_dim: int,
        nsamples: Mapping[str, int],
        nfeatures: int,
        *,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        **kwargs,
    ) -> PyroLikelihood:
        return PyroNormal(
            self._view_name,
            sample_dim,
            feature_dim,
            nsamples,
            nfeatures,
            self._shift,
            self._scale,
            init_scale=init_scale,
        )

    @classmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        return True

    @classmethod
    def _r2(
        cls,
        r2_full: float,
        y_true: NDArray,
        factors: NDArray[np.floating],
        weights: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        # this is the same as MOFA2
        r2s = np.empty(factors.shape[1], dtype=np.float32)
        for k in range(factors.shape[1]):
            r2s[k] = cls._r2_impl_wrapper(y_true, factors[:, k, None], weights[:, k, None], dispersions, sample_means)
        return r2s

    @classmethod
    def _r2_impl(
        cls,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> R2:
        ss_res = np.nansum(np.square(y_true - y_pred))
        ss_tot = np.nansum(np.square(y_true))  # data is centered
        return R2(ss_res, ss_tot)

    def transform_prediction(self, prediction: NDArray[np.floating], group_name: str):
        return prediction + self._shift[group_name]
