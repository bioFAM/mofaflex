from collections.abc import Mapping
from contextlib import suppress
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from array_api_compat import array_namespace

from ..datasets import MofaFlexDataset, merge_covariates
from ..utils import Matrix, MeanStd, Vector, nanmean, nanmin, nanvar
from .base import R2, Likelihood
from .pyro import Likelihood as PyroLikelihood
from .pyro import Normal as PyroNormal


class Normal(Likelihood):
    """Gaussian likelihood for continuous data.

    Args:
        scale_per_group: Scale data per group, otherwise across all groups.
        stddev_var_key: The column of `.var` that contains known feature-wise standard deviations. If `None`, standard deviations
            will be infered during training. If `scale_per_group=False`, the feature-wise standard deviation will be averaged
            over groups.
    """

    _priority = 0
    _state_attrs = ("_stddev_var_key", "_shift", "_scale", "_dispersion")

    def __init__(
        self,
        view_name: str,
        data: MofaFlexDataset,
        nonnegative: bool,
        scale_per_group: bool = True,
        stddev_var_key: str | None = None,
    ):
        super().__init__(view_name, data, nonnegative)
        self._scale_per_group = scale_per_group
        self._stddev_var_key = stddev_var_key

        statfun = nanmean if not nonnegative else nanmin
        self._shift = data.apply_to_view(view_name, lambda adata, group_name: statfun(adata.X, axis=0))

        if stddev_var_key is not None:
            scale = data.get_covariates(1, stddev_var_key, filter_names=view_name)
            if scale_per_group:
                self._dispersion = {
                    group_name: group.to_numpy().squeeze() for group_name, group in scale[view_name].items()
                }
            else:
                self._dispersion = merge_covariates(scale)[view_name].to_numpy().squeeze()
            self._scale = None
        else:
            self._dispersion = None
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

    def _calc_scale_ungrouped(self, adata: AnnData, group: Vector[str], view_name: str, groups: list[str]):
        if adata.n_obs <= 1:
            return 1.0

        arr = adata.X.copy()
        for group_name in groups:
            arr[group == group_name] -= align_local_array_to_global(  # noqa F821
                self._shift[group_name], group_name, view_name, align_to="features", axis=0
            )
        return np.sqrt(nanvar(arr, axis=None))

    def _calc_scale_grouped(self, adata: AnnData, group_name: str):
        arr = adata.X - np.broadcast_to(
            self._shift[group_name], adata.X.shape
        )  # need to manually broadcast to force sparse to autoconvert to dense instead of raising
        if isinstance(arr, np.matrix):
            arr = np.asarray(arr)
        arr = nanvar(arr, axis=None)
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
            dispersion=self._dispersion,
            init_scale=init_scale,
        )

    def on_train_end(self, *args, **kwargs):
        self._dispersion = self._pyro_likelihood.dispersion

    @classmethod
    def _validate(cls, data: Matrix[np.number], xp) -> bool:
        return True

    def _r2_impl(
        self,
        y_true: Matrix[np.number],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> R2:
        ss_res = np.nansum(np.square(y_true - y_pred))
        ss_tot = np.nansum(np.square(y_true - self._shift[group_name][feature_idx]))
        return R2(ss_res, ss_tot)

    def _deviance_explained_impl(
        self,
        y_true: Matrix[np.number],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> R2:
        # fraction of deviance explained reduces to standard R2 for Gaussian likelihood
        return self._r2_impl(
            y_true,
            self.transform_prediction(y_pred, group_name, sample_idx, feature_idx),
            group_name,
            sample_idx,
            feature_idx,
        )

    def transform_prediction(
        self,
        prediction: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating]:
        if (scale := self._scale) is not None:
            with suppress(IndexError):
                scale = self._scale[group_name]
            prediction = prediction * scale
        return prediction + self._shift[group_name][feature_idx]

    def transform_data(
        self,
        data: Matrix[np.number],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.number]:
        transformed = data - self._shift[group_name][feature_idx]
        if (scale := self._scale) is not None:
            with suppress(IndexError):
                scale = scale[group_name]
            transformed /= scale
        return transformed

    @Likelihood._api
    def get_dispersion(self, moment: Literal["mean", "std"] = "mean") -> pd.Series | dict[str, pd.Series]:
        """Get the dispersion vectors for each view.

        Args:
            moment: Which moment of the posterior distribution to return. Ignored if the likelihood was instantiated with `stddev_var_key`.

        Returns:
            If the likelhood was instantiated with `stddev_var_key`, a :class:`~pandas.Series` if `scale_per_group=False`, otherwise a
            dictionary of :class:`~pandas.Series`, one per group.

            Otherwise a :class:`~pandas.Series` containing the requested moment of the inferred posterior distribution.
        """
        if isinstance(self._dispersion, MeanStd):
            return pd.Series(getattr(self._dispersion, moment), index=self._feature_names)
        elif isinstance(self._dispersion, Mapping):
            return {
                group_name: pd.Series(disp, index=self._feature_names) for group_name, disp in self._dispersion.items()
            }
        else:
            return pd.Series(self._dispersion, index=self._feature_names)
