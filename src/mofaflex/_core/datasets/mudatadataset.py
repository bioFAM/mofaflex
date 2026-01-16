import logging
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, TypeVar, Union

import anndata as ad
import numpy as np
import pandas as pd
from mudata import MuData
from numpy.typing import NDArray
from scipy import sparse

from ..settings import settings
from .base import ApplyCallable, ApplyToCallable, MofaFlexDataset, Preprocessor
from .utils import (
    align_dataframe,
    anndata_to_dask,
    apply_to_nested,
    array_to_dask,
    from_dask,
    have_dask,
    select_anndata_layer,
    warn_dask,
)

T = TypeVar("T")
_logger = logging.getLogger(__name__)


def _fixup_mudata(mudata: MuData, orig: MuData, with_extra: bool = True, extra_callback=lambda x: x):
    # MuData constructor runs update(), so we need to reset obs and var
    mudata.obs = orig.obs
    mudata.var = orig.var
    if with_extra:
        for attrname in ("obsm", "obsp", "varm", "varp"):
            attr = getattr(orig, attrname)
            new_attr = getattr(mudata, attrname)
            for k, v in attr.items():
                new_attr[k] = extra_callback(v)
    return mudata


def _mudata_to_dask(mudata: MuData, with_extra: bool = True):
    mods = {modname: anndata_to_dask(mod) for modname, mod in mudata.mod.items()}
    dask_mudata = MuData(mods, obs=mudata.obs, var=mudata.var, obsmap=mudata.obsmap, varmap=mudata.varmap)
    return _fixup_mudata(dask_mudata, mudata, with_extra=with_extra, extra_callback=array_to_dask)


def _select_layers(mudata: MuData, layer: Mapping[str, str | None] | None):
    if layer is None:
        return mudata

    if isinstance(layer, str):
        layerfunc = lambda modname: layer
    else:
        layerfunc = lambda modname: layer.get(modname)

    new_mudata = MuData(
        {modname: select_anndata_layer(mod, layerfunc(modname)) for modname, mod in mudata.mod.items()},
        obs=mudata.obs,
        var=mudata.var,
        obsmap=mudata.obsmap,
        varmap=mudata.varmap,
    )
    return _fixup_mudata(new_mudata, mudata, with_extra=True)


class MuDataDataset(MofaFlexDataset):
    def __init__(
        self,
        mudata: MuData,
        *,
        layer: Mapping[str, str | None] | str | None = None,
        group_by: str | Sequence[str] | None = None,
        preprocessor: Preprocessor | None = None,
        cast_to: np.number | None = np.float32,
        subset_var: str | None = "highly_variable",
        sample_names: Mapping[str, NDArray[str]] | None = None,
        feature_names: Mapping[str, NDArray[str]] | None = None,
        **kwargs,
    ):
        # MuData objects with axis=1 can violate our fundamendal data model of local and global alignment:
        # Since group assignments are taken from the global .obs, the same group may be present in multiple
        # AnnData objects with only partially overlapping features. In this case it is not clear what the
        # local samples and features should be.
        if mudata.axis != 0:
            raise ValueError("Only MuData objects with axis=0 are supported.")

        super().__init__(mudata, preprocessor=preprocessor, cast_to=cast_to)
        self._orig_data = _select_layers(self._data, layer)
        self._group_by = group_by
        self._sample_selection = self._feature_selection = slice(None)
        self._groups = None

        if feature_names is None and subset_var is not None:
            feature_names = {}
            if subset_var in self._orig_data.var:
                for modname, modmap in self._orig_data.varmap.items():
                    feature_names[modname] = self._orig_data.mod[modname].var_names[
                        self._orig_data.var[subset_var][modmap.reshape(-1) > 0]
                    ]
            else:
                for modname, mod in self._orig_data.mod.items():
                    if subset_var in mod.var:
                        feature_names[modname] = mod.var_names[mod.var[subset_var]]

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

        self._needs_alignment = {}
        for group_name, group_idx in self._groups.items():
            gneeds_align = set()
            for view_name, obsmap in self._data.obsmap.items():
                obsmap = obsmap[group_idx]
                if np.any(obsmap == 0) or np.any(np.diff(obsmap) != 1):
                    gneeds_align.add(view_name)
            self._needs_alignment[group_name] = gneeds_align

    def _get_groups(self, df):
        return df.groupby(
            pd.Categorical(df[self._group_by]).rename_categories(lambda x: str(x))
            if self._group_by is not None
            else lambda x: "group_1",
            observed=True,
        ).indices

    def reindex_features(self, feature_names: Mapping[str, NDArray[str]] | None = None):
        if feature_names is not None and any(
            feature_names[view_name].size != fnames.size or np.any(feature_names[view_name] != fnames)
            for view_name, fnames in self.feature_names.items()
            if view_name in feature_names
        ):
            selection = pd.Index(())
            for modname, mod in self._orig_data.mod.items():
                view_feature_names = feature_names.get(modname)
                if view_feature_names is not None:
                    view_feature_names = pd.Index(view_feature_names)
                    if np.any(~view_feature_names.isin(mod.var_names)):
                        _logger.warning(
                            f"Not all feature names given for view {modname} are present in the data. Restricting alignment to feature names present in the data."
                        )
                        view_feature_names = view_feature_names.intersection(mod.var_names)
                else:
                    view_feature_names = mod.var_names
                selection = selection.append(view_feature_names)
            self._data = self._orig_data[self._sample_selection, selection]
            self._feature_selection = selection
        elif feature_names is None:
            self._data = self._orig_data[self._sample_selection, :]
            self._feature_selection = slice(None)

    @staticmethod
    def _accepts_input(data):
        return isinstance(data, MuData)

    @property
    def n_features(self) -> dict[str, int]:
        return {modname: mod.n_vars for modname, mod in self._data.mod.items()}

    @property
    def n_samples(self) -> dict[str, int]:
        return {groupname: len(groupidx) for groupname, groupidx in self._groups.items()}

    @property
    def n_samples_total(self) -> int:
        return self._data.n_obs

    @property
    def view_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._data.mod.keys()))

    @property
    def group_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._groups.keys()))

    @property
    def sample_names(self) -> dict[str, NDArray[str]]:
        return {groupname: self._data[groupidx, :].obs_names.to_numpy() for groupname, groupidx in self._groups.items()}

    @property
    def feature_names(self) -> dict[str, NDArray[str]]:
        return {viewname: mod.var_names.to_numpy() for viewname, mod in self._data.mod.items()}

    def __getitems__(self, idx: Mapping[str, int | Sequence[int]]) -> dict[str, dict]:
        data = {}
        nonmissing_obs = {}
        nonmissing_var = {}
        for group_name, group_idx in idx.items():
            group = {}
            gnonmissing_obs = {}
            gnonmissing_var = {}
            glabel = self._groups[group_name][group_idx]
            subdata = self._data[glabel, :]
            for modname, mod in subdata.mod.items():
                cnonmissing_obs = (
                    np.nonzero(subdata.obsmap[modname] > 0)[0]
                    if modname in self._needs_alignment[group_name]
                    else slice(None)
                )
                arr, gnonmissing_obs[modname], gnonmissing_var[modname] = self.preprocessor(
                    mod.X, cnonmissing_obs, slice(None), group_name, modname
                )
                if self.cast_to is not None:
                    arr = arr.astype(self.cast_to, copy=False)
                if sparse.issparse(arr):
                    arr = arr.toarray()
                group[modname] = np.asarray(
                    arr
                )  # arr may be an anndata._core.views.ArrayView, which is not recognized by PyTorch
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

    def _align_local_array_to_global(
        self,
        arr: NDArray[T],
        view_name: str,
        subdata: MuData | None = None,
        group_name: str | None = None,
        align_to: Literal["samples", "features"] = "samples",
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ) -> NDArray[T]:
        if align_to == "features":
            return arr

        if subdata is None:
            if group_name is None:
                raise ValueError("Need either subdata or group_name, but both are None.")
            if view_name not in self._needs_alignment[group_name]:
                return arr
            subdata = self._data[self._groups[group_name], :]

        viewidx = subdata.obsmap[view_name]
        nnz = viewidx > 0

        outshape = [subdata.n_obs] + list(arr.shape[:axis]) + list(arr.shape[axis + 1 :])

        out = np.full(outshape, fill_value=fill_value, dtype=np.promote_types(type(fill_value), arr.dtype), order="C")
        out[nnz, ...] = np.moveaxis(arr, axis, 0)[viewidx[nnz] - 1, ...]
        return np.moveaxis(out, 0, axis)

    def align_local_array_to_global(
        self,
        arr: NDArray[T],
        group_name: str,
        view_name: str,
        align_to: Literal["samples", "features"],
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ):
        return self._align_local_array_to_global(
            arr, view_name, group_name=group_name, align_to=align_to, axis=axis, fill_value=fill_value
        )

    def align_global_array_to_local(
        self, arr: NDArray[T], group_name: str, view_name: str, align_to: Literal["samples", "features"], axis: int = 0
    ) -> NDArray[T]:
        if align_to == "samples":
            idx = self.map_local_indices_to_global(slice(None), group_name, view_name, align_to)
            return np.take(arr, idx, axis=axis)
        else:
            return arr

    def map_local_indices_to_global(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal["samples, features"]
    ) -> NDArray[int]:
        if align_to == "samples":
            subdata = self._data[self._groups[group_name], :]
            map = subdata.obsmap[view_name]
            mask = map > 0
            n = subdata.mod[view_name].n_obs
            viewidx = np.empty(n, dtype=map.dtype)
            viewidx[map[mask] - 1] = np.arange(n, dtype=map.dtype) + np.cumsum(~mask)[mask]

            return viewidx[idx]
        else:
            return idx

    def map_global_indices_to_local(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal["samples, features"]
    ) -> NDArray[int]:
        if align_to == "samples":
            subdata = self._data[self._groups[group_name], :]
            return subdata.obsmap[view_name][idx].astype(int) - 1
        else:
            return idx

    def get_obs(self) -> dict[str, pd.DataFrame]:
        # We don't want to duplicate MuData's push_obs logic, but at the same time
        # we don't want to modify the data object. So we create a temporary fake
        # MuData object with the same metadata, but no actual data
        fakeadatas = {
            modname: ad.AnnData(X=sparse.csr_array(mod.X.shape), obs=mod.obs, var=mod.var)
            for modname, mod in self._data.mod.items()
        }

        # need to pass obs in the constructor to make shape validation for obsmap work
        fakemudata = MuData(fakeadatas, obs=self._data.obs, obsmap=self._data.obsmap)
        # need to replace obs since the constructor runs update(), which breaks push_obs()
        fakemudata.obs = self._data.obs
        fakemudata.push_obs()
        return {
            group_name: {
                modname: mod.obs.reindex(self._data[group_idx, :].obs_names, fill_value=pd.NA).apply(
                    lambda x: x.astype("string") if x.dtype == "O" else x, axis=1
                )
                for modname, mod in fakemudata.mod.items()
            }
            for group_name, group_idx in self._groups.items()
        }

    def get_missing_obs(self) -> pd.DataFrame:
        dfs = []
        for group_name, group_idx in self._groups.items():
            subdata = self._data[group_idx, :]
            for modname, mod in subdata.mod.items():
                if sparse.issparse(mod.X):
                    modmissing = mod.X.copy()
                    modmissing.data = np.isnan(modmissing.data)
                    modmissing = ~(np.asarray(modmissing.sum(axis=1)).squeeze() == 0)
                else:
                    modmissing = np.isnan(mod.X).all(axis=1)
                modmissing = self._align_local_array_to_global(modmissing, modname, subdata, fill_value=True)
                dfs.append(
                    pd.DataFrame(
                        {"view": modname, "group": group_name, "obs_name": subdata.obs_names, "missing": modmissing}
                    )
                )
        return pd.concat(dfs, axis=0, ignore_index=True)

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
        covar_dims = defaultdict(set)
        for group_name, group_idx in self._groups.items():
            if axis == 0 and filter_names is not None and group_name not in filter_names:
                continue
            for modname in self.view_names:
                if axis == 1 and filter_names is not None and modname not in filter_names:
                    continue
                subdata = self._data[group_idx, self.feature_names[modname]]
                mod = subdata.mod[modname]
                outer_key, inner_key = (group_name, modname)[dict_reorder]

                ckey = key.get(outer_key, None)
                cmkey = mkey.get(outer_key, None)

                if ckey is None and cmkey is None:
                    continue
                if ckey and cmkey:
                    raise ValueError(f"Provide either key or mkey for {outer_msg} {outer_key}, not both.")

                ccov = None
                if ckey is not None:
                    if ckey in getattr(mod, attr).columns:
                        ccov = align_dataframe(getattr(mod, attr)[[ckey]], getattr(subdata, attrnames))
                    elif ckey in getattr(subdata, attr).columns:
                        ccov = getattr(subdata, attr)[[ckey]]
                    if ccov is not None:
                        covariates[outer_key][inner_key] = ccov
                elif cmkey is not None:
                    needs_alignment = False
                    if cmkey in getattr(mod, attrm):
                        ccov = getattr(mod, attrm)[cmkey]
                        needs_alignment = True
                    elif cmkey in getattr(subdata, attrm):
                        ccov = getattr(subdata, attrm)[cmkey]
                    if ccov is not None:
                        if sparse.issparse(ccov):
                            ccov = ccov.toarray()
                        if isinstance(ccov, pd.Series):
                            if not ccov.name:
                                ccov.name = cmkey
                            ccov = pd.DataFrame(ccov)
                        elif isinstance(ccov, np.ndarray):
                            ccov = pd.DataFrame(
                                ccov,
                                index=getattr(subdata, attrnames) if not needs_alignment else getattr(mod, attrnames),
                            )
                            if ccov.shape[1] == 1:
                                ccov.columns = [cmkey]
                        covar_dims[outer_key].add(ccov.shape[1])

                        if needs_alignment:
                            ccov = align_dataframe(ccov, getattr(subdata, attrnames), fill_value=fill_value)
                        covariates[outer_key][inner_key] = ccov

        for name, covar_dim in covar_dims.items():
            if len(covar_dim) > 1:
                raise ValueError(
                    f"Number of covariate dimensions in {outer_msg} {name} must be the same across {inner_msg}s."
                )
        return dict(covariates)

    def _data_for_apply(self):
        data = self._data
        if settings.use_dask:
            if have_dask():
                data = _mudata_to_dask(self._orig_data, with_extra=False)[
                    self._sample_selection, self._feature_selection
                ]
            else:
                warn_dask(_logger)
        return data

    def _apply_to_view(
        self, view_name: str, func: ApplyToCallable[T], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data_for_apply()
        ret = {}
        for group_name, group_idx in self._groups.items():
            cret = func(data[group_idx, :][view_name], group_name, **kwargs, **gkwargs[group_name])
            ret[group_name] = apply_to_nested(cret, from_dask)
        return ret

    def _apply_to_group(
        self, group_name: str, func: ApplyToCallable[T], vkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data_for_apply()
        ret = {}
        data = data[self._groups[group_name], :]
        for modname, mod in data.mod.items():
            cret = func(mod, modname, **kwargs, **vkwargs[modname])
            ret[modname] = apply_to_nested(cret, from_dask)
        return ret

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
            cret = {}
            subdata = data[group_idx, :]
            for modname in view_names:
                ccret = func(subdata.mod[modname], group_name, modname, **kwargs, **gvkwargs[group_name][modname])
                cret[modname] = apply_to_nested(ccret, from_dask)
            ret[group_name] = cret
        return ret

    def _apply_by_view(
        self, func: ApplyCallable[T], view_names: Sequence[str], vkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data
        if (
            not isinstance(self._sample_selection, slice)
            or self._sample_selection != slice(None)
            or not isinstance(self._feature_selection, slice)
            or self._feature_selection != slice(None)
        ) and settings.use_dask:
            if have_dask():
                data = _mudata_to_dask(self._orig_data, with_extra=False)[
                    self._sample_selection, self._feature_selection
                ]
            else:
                warn_dask(_logger)
        ret = {}
        for modname in view_names:
            mod = data.mod[modname]
            groups = np.empty((mod.n_obs,), dtype="O")
            for group, group_idx in self._groups.items():
                modidx = self._data.obsmap[modname][group_idx]
                modidx = modidx[modidx > 0] - 1
                groups[modidx] = group

            cret = func(mod, groups, modname, **kwargs, **vkwargs[modname])
            ret[modname] = apply_to_nested(cret, from_dask)
        return ret

    def _apply_by_group(
        self, func: ApplyCallable[T], group_names: Sequence[str], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data_for_apply()
        ret = {}
        for group_name in group_names:
            group_idx = self._groups[group_name]
            subdata = data[group_idx, :]
            gdata = {}
            convert = False
            for modname, mod in subdata.mod.items():
                if mod.n_obs != subdata.n_obs:
                    convert = True
                gdata[modname] = mod
            if convert:
                for modname, mod in gdata.items():
                    mod = mod.copy()
                    mod.X = mod.X.astype(np.promote_types(mod.X.dtype, type(np.nan)))
                    gdata[modname] = mod
            gdata = ad.concat(
                gdata, axis="var", join="outer", label="__view", merge="unique", uns_merge=None, fill_value=np.nan
            )
            if (gdata.obs_names != subdata.obs_names).any():
                gdata = gdata[subdata.obs_names, :]
            cret = func(gdata, group_name, gdata.var["__view"].to_numpy(), **kwargs, **gkwargs[group_name])
            ret[group_name] = apply_to_nested(cret, from_dask)
        return ret
