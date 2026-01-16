import numpy as np
from anndata import AnnData
from array_api_compat import array_namespace
from numpy.typing import NDArray

from .. import utils
from ..datasets import MofaFlexDataset
from .base import R2, Likelihood
from .pyro import Likelihood as PyroLikelihood
from .pyro import Normal as PyroNormal


class Normal(Likelihood):
    """Gaussian likelihood for continuous data.

    Args:
        scale_per_group: Scale data per group, otherwise across all groups.
    """

    _priority = 0
    _state_attrs = ("_shift", "_scale", "_dispersion")

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool, scale_per_group: bool = True):
        super().__init__(view_name, data, nonnegative)
        self._scale_per_group = scale_per_group
        statfun = utils.nanmean if not nonnegative else utils.nanmin
        self._shift = data.apply_to_view(view_name, lambda adata, group_name: statfun(adata.X, axis=0))

        if scale_per_group:
            self._scale = data.apply_to_view(view_name, self._calc_scale_grouped)
        else:
            self._scale = data.apply(
                self._calc_scale_ungrouped, by_group=False, filter_views=view_name, groups=data.group_names
            )[view_name]
        self._shift = {
            group_name: data.align_local_array_to_global(shift, group_name, self._view_name, align_to="features")
            for group_name, shift in self._shift.items()
        }

        self._dispersion = None

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

    def _get_pyro_likelihood(
        self,
        data: MofaFlexDataset,
        sample_dim: int,
        feature_dim: int,
        *,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
    ) -> PyroLikelihood:
        return PyroNormal(
            self._view_name,
            sample_dim,
            feature_dim,
            data.n_samples,
            data.n_features[self._view_name],
            shift=self._shift,
            scale=self._scale,
            init_scale=init_scale,
        )

    def on_train_end(self, *args, **kwargs):
        self._dispersion = self._pyro_likelihood.dispersion

    @classmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        return True

    def _r2_impl(
        self,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ) -> R2:
        ss_res = np.nansum(np.square(y_true - y_pred))
        ss_tot = np.nansum(np.square(y_true - self._shift[group_name][feature_idx]))
        return R2(ss_res, ss_tot)

    def transform_prediction(
        self,
        prediction: NDArray[np.floating],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ):
        try:
            scale = self._scale[group_name]
        except IndexError:
            scale = self._scale
        return prediction * scale + self._shift[group_name][feature_idx]

    def transform_data(
        self,
        data: NDArray[np.floating],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ):
        try:
            scale = self._scale[group_name]
        except IndexError:
            scale = self._scale
        return data - self._shift[group_name][feature_idx] / scale

    @property
    def dispersion(self):
        return self._dispersion
