import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeVar, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import issparse

from ..settings import settings
from .base import ApplyCallable, ApplyToCallable, MofaFlexDataset, Preprocessor
from .utils import anndata_to_dask, apply_to_nested, from_dask, have_dask, select_anndata_layer, warn_dask

T = TypeVar("T")
_logger = logging.getLogger(__name__)


class AnnDataDataset(MofaFlexDataset):
    _view_name = "view_1"

    def __init__(
        self,
        adata: AnnData,
        *,
        layer: str | None = None,
        group_by: str | Sequence[str] | None = None,
        preprocessor: Preprocessor | None = None,
        cast_to: np.number | None = np.float32,
        subset_var: str | None = "highly_variable",
        sample_names: Mapping[str, NDArray[str]] | None = None,
        feature_names: Mapping[str, NDArray[str]] | None = None,
        **kwargs,
    ):
        super().__init__(adata, preprocessor=preprocessor, cast_to=cast_to)
        self._orig_data = select_anndata_layer(self._data, layer)
        self._group_by = group_by
        self._sample_selection = self._feature_selection = slice(None)
        self._groups = None

        if feature_names is None and subset_var is not None and subset_var in self._orig_data.var:
            feature_names = {self._view_name: self._orig_data.var_names[self._orig_data.var[subset_var]]}

        self.reindex_samples(sample_names)
        self.reindex_features(feature_names)

    def reindex_samples(self, sample_names: Mapping[str, NDArray[str]] | None = None):
        if sample_names is not None and (
            self._groups is None
            or any(
                sample_names[group_name].size != group_idx.size
                or np.any(sample_names[group_name] != self._data.obs_names[group_idx])
                for group_name, group_idx in self._groups.items()
                if group_name in sample_names
            )
        ):
            groups = self._get_groups(self._orig_data.obs)
            selection = pd.Index(())
            for group_name, group_idx in groups.items():
                group_sample_names = sample_names.get(group_name)
                if group_sample_names is not None:
                    group_sample_names = pd.Index(group_sample_names)
                    if np.any(~group_sample_names.isin(self._orig_data.obs_names[group_idx])):
                        _logger.warning(
                            f"Not all sample names given for group {group_name} are present in the data. Restricting alignment to group names present in the data."
                        )
                        group_sample_names = group_sample_names.intersection(self._orig_data.obs_names[group_idx])
                else:
                    group_sample_names = self._orig_data.obs_names[group_idx]
                selection = selection.append(group_sample_names)
            self._data = self._orig_data[selection, self._feature_selection]
            self._sample_selection = selection
        elif sample_names is None:
            self._data = self._orig_data[:, self._feature_selection]
            self._sample_selection = slice(None)

        self._groups = self._get_groups(self._data.obs)

    def _get_groups(self, df):
        return df.groupby(
            pd.Categorical(df[self._group_by]).rename_categories(lambda x: str(x))
            if self._group_by is not None
            else lambda x: "group_1",
            observed=True,
        ).indices

    def reindex_features(self, feature_names: Mapping[str, NDArray[str]] | None = None):
        if (
            feature_names is not None
            and self._view_name in feature_names
            and (
                (view_feature_names := feature_names[self._view_name]).size != self._data.n_vars
                or np.any(view_feature_names != self._data.var_names)
            )
        ):
            view_feature_names = pd.Index(view_feature_names)
            if np.any(~view_feature_names.isin(self._orig_data.var_names)):
                _logger.warning(
                    "Not all feature names given are present in the data. Restricting alignment to feature names present in the data."
                )
                selection = view_feature_names.intersection(self._orig_data.var_names)
            else:
                selection = view_feature_names
            self._data = self._orig_data[self._sample_selection, selection]
            self._feature_selection = selection
        elif feature_names is None:
            self._data = self._orig_data[self._sample_selection, :]
            self._feature_selection = slice(None)

    @staticmethod
    def _accepts_input(data):
        return isinstance(data, AnnData)

    @property
    def n_features(self) -> dict[str, int]:
        return {self._view_name: self._data.n_vars}

    @property
    def n_samples(self) -> dict[str, int]:
        return {groupname: len(groupidx) for groupname, groupidx in self._groups.items()}

    @property
    def n_samples_total(self) -> int:
        return self._data.n_obs

    @property
    def view_names(self) -> NDArray[str]:
        return np.asarray((self._view_name,))

    @property
    def group_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._groups.keys()))

    @property
    def sample_names(self) -> dict[str, NDArray[str]]:
        return {groupname: self._data.obs_names[groupidx].to_numpy() for groupname, groupidx in self._groups.items()}

    @property
    def feature_names(self) -> dict[str, NDArray[str]]:
        return {self._view_name: self._data.var_names.to_numpy()}

    def __getitems__(self, idx: Mapping[str, int | Sequence[int]]) -> dict[str, dict]:
        data = {}
        nonmissing_obs = {}
        nonmissing_var = {}
        for group_name, group_idx in idx.items():
            gnonmissing_obs = {}
            gnonmissing_var = {}
            glabel = self._groups[group_name][group_idx]
            subdata = self._data[glabel, :]
            arr, gnonmissing_obs[self._view_name], gnonmissing_var[self._view_name] = self.preprocessor(
                subdata.X, slice(None), slice(None), group_name, self._view_name
            )
            if self.cast_to is not None:
                arr = arr.astype(self.cast_to, copy=False)
            if issparse(arr):
                arr = arr.toarray()
            group = {
                self._view_name: np.asarray(arr)
            }  # arr may be an anndata._core.views.ArrayView, which is not recognized by PyTorch
            data[group_name] = group
            idx[group_name] = np.asarray(group_idx)
            nonmissing_obs[group_name] = gnonmissing_obs
            nonmissing_var[group_name] = gnonmissing_var
        return {
            "data": data,
            "sample_idx": idx,
            "nonmissing_samples": nonmissing_obs,
            "nonmissing_features": nonmissing_var,
        }

    def align_local_array_to_global(
        self,
        arr: NDArray[T],
        group_name: str,
        view_name: str,
        align_to: Literal["samples", "features"],
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ):
        return arr

    def align_global_array_to_local(
        self, arr: NDArray[T], group_name: str, view_name: str, align_to: Literal["samples", "features"], axis: int = 0
    ) -> NDArray[T]:
        return arr

    def map_local_indices_to_global(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal["samples, features"]
    ) -> NDArray[int]:
        return idx

    def map_global_indices_to_local(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal["samples, features"]
    ) -> NDArray[int]:
        return idx

    def get_obs(self) -> dict[str, pd.DataFrame]:
        return {
            group_name: {
                self._view_name: self._data[group_idx, :].obs.apply(
                    lambda x: x.astype("string") if x.dtype == "O" else x, axis=1
                )
            }
            for group_name, group_idx in self._groups.items()
        }

    def get_missing_obs(self) -> pd.DataFrame:
        if issparse(self._data.X):
            missing = self._data.X.copy()
            missing.data = np.isnan(missing.data)
            missing = np.asarray(missing.sum(axis=1)).squeeze() == missing.shape[1]
        else:
            missing = np.isnan(self._data.X).all(axis=1)
        df = pd.DataFrame({"view": self._view_name, "group": "", "obs_name": "", "missing": missing})
        for group_name, group_idx in self._groups.items():
            df.loc[df.index[group_idx], "group"] = group_name
            df.loc[df.index[group_idx], "obs_name"] = self._data.obs_names[group_idx]
        return df

    def _get_covariates(
        self,
        axis: int,
        key: Mapping[str, str],
        mkey: Mapping[str, str],
        filter_names: Sequence[str] | None,
        fill_value: Callable[[np.dtype | pd.api.extensions.ExtensionDtype], Union[*np.ScalarType]],
    ) -> dict[str, dict[str, pd.DataFrame]]:
        if axis == 0:
            attr = "obs"
            dict_reorder = slice(None)
        else:
            attr = "var"
            dict_reorder = slice(None, None, -1)
        attrm = f"{attr}m"
        attrnames = f"{attr}_names"
        outer_msg, inner_msg = ("group", "view")[dict_reorder]

        covariates = defaultdict(dict)
        for group_name, group_idx in self._groups.items():
            if axis == 0 and filter_names is not None and group_name not in filter_names:
                continue
            if axis == 1 and filter_names is not None and self._view_name not in filter_names:
                continue
            subdata = self._data[group_idx, :]
            outer_key, inner_key = (group_name, self._view_name)[dict_reorder]

            ckey = key.get(outer_key, None)
            cmkey = mkey.get(outer_key, None)
            if ckey is None and cmkey is None:
                continue
            if ckey and cmkey:
                raise ValueError(
                    f"Provide either key or mkey for {outer_msg} {outer_key}, got key='{ckey}', mkey='{cmkey}'."
                )

            if ckey is not None:
                if ckey in getattr(subdata, attr).columns:
                    ccov = getattr(subdata, attr)[[ckey]]
                    covariates[outer_key][inner_key] = ccov
            elif cmkey is not None:
                if cmkey in getattr(subdata, attrm):
                    ccov = getattr(subdata, attrm)[cmkey]
                    if issparse(ccov):
                        ccov = ccov.toarray()
                    if isinstance(ccov, pd.Series):
                        if not ccov.name:
                            ccov.name = cmkey
                        ccov = pd.DataFrame(ccov)
                    elif isinstance(ccov, np.ndarray):
                        ccov = pd.DataFrame(ccov, index=getattr(subdata, attrnames))
                        if ccov.shape[1] == 1:
                            ccov.columns = [cmkey]
                    covariates[outer_key][inner_key] = ccov
        return dict(covariates)

    def _data_for_apply(self):
        data = self._data
        if settings.use_dask:
            if have_dask():
                data = anndata_to_dask(self._orig_data)[self._sample_selection, self._feature_selection]
            else:
                warn_dask(_logger)
        return data

    def _apply_to_view(
        self, view_name: str, func: ApplyToCallable[T], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data_for_apply()
        ret = {}
        for group_name, group_idx in self._groups.items():
            cret = func(data[group_idx, :], group_name, **kwargs, **gkwargs[group_name])
            ret[group_name] = apply_to_nested(cret, from_dask)
        return ret

    def _apply_to_group(
        self, group_name: str, func: ApplyToCallable[T], vkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data_for_apply()
        data = data[self._groups[group_name], :]
        return {
            self._view_name: apply_to_nested(
                func(data, self._view_name, **kwargs, **vkwargs[self._view_name]), from_dask
            )
        }

    def _apply_by_group_view(
        self,
        func: ApplyCallable[T],
        group_names: Sequence[str],
        view_names: Sequence[str],
        gvkwargs: Mapping[str, Mapping[str, Mapping[str, Any]]],
        **kwargs,
    ) -> dict[str, dict[str, T]]:
        data = self._data_for_apply()
        ret = {}
        for group_name in group_names:
            group_idx = self._groups[group_name]
            ret[group_name] = {
                self._view_name: apply_to_nested(
                    func(
                        data[group_idx, :],
                        group_name,
                        self._view_name,
                        **kwargs,
                        **gvkwargs[group_name][self._view_name],
                    ),
                    from_dask,
                )
            }
        return ret

    def _apply_by_view(
        self, func: ApplyCallable[T], view_names: Sequence[str], vkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        groups = np.empty((self._data.n_obs,), dtype="O")
        for group_name, group_idx in self._groups.items():
            groups[group_idx] = group_name
        return {self._view_name: func(self._data, groups, self._view_name, **kwargs, **vkwargs[self._view_name])}

    def _apply_by_group(
        self, func: ApplyCallable[T], group_names: Sequence[str], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data_for_apply()
        ret = {}
        for group_name in group_names:
            group_idx = self._groups[group_name]
            subdata = data[group_idx, :]
            ret[group_name] = apply_to_nested(
                func(subdata, group_name, np.broadcast_to(self._view_name, (subdata.n_obs,))), from_dask
            )
        return ret
