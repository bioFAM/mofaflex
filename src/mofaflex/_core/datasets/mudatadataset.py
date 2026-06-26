import logging
from abc import abstractmethod
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
    dask_mudata = MuData(
        mods, obs=mudata.obs, var=mudata.var, obsmap=mudata.obsmap, varmap=mudata.varmap, axis=mudata.axis
    )
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
        axis=mudata.axis,
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
        groups: Mapping[str, NDArray[int]] | None = None,
        **kwargs,
    ):
        super().__init__(mudata, preprocessor=preprocessor, cast_to=cast_to)
        self._orig_data = _select_layers(self._data, layer)
        self._group_by = group_by
        self._sample_selection = self._feature_selection = slice(None)
        self._groups = None
        self._orig_groups = groups

        self.reindex_samples(sample_names)
        self.reindex_features(feature_names)

    def _reindex_with_groups(
        self, attr: str, names: Mapping[str, NDArray[str]] | None = None
    ) -> pd.Index | slice | None:
        namesattr = f"{attr}_names"
        if names is not None and (
            self._groups is None
            or any(
                names[group_name].size != group_idx.size
                or np.any(names[group_name] != getattr(self._data, namesattr)[group_idx])
                for group_name, group_idx in self._groups.items()
                if group_name in names
            )
        ):
            groups = (
                self._get_groups(getattr(self._orig_data, attr)) if self._orig_groups is None else self._orig_groups
            )
            selection = pd.Index(())
            for group_name, group_idx in groups.items():
                group_names = names.get(group_name)
                if group_names is not None:
                    group_names = pd.Index(group_names)
                    if np.any(~group_names.isin(getattr(self._orig_data, namesattr)[group_idx])):
                        _logger.warning(
                            f"Not all names given for group {group_name} are present in the data. Restricting alignment to group names present in the data."
                        )
                        group_names = group_names.intersection(getattr(self._orig_data, namesattr)[group_idx])
                else:
                    group_names = getattr(self._orig_data, namesattr)[group_idx]
                selection = selection.append(group_names)
        elif names is None:
            selection = slice(None)
        else:
            selection = None

        return selection

    def _calc_groups(self, attr: str):
        mapattr = f"{attr}map"

        self._groups = self._get_groups(getattr(self._data, attr))
        self._needs_alignment = {}
        for group_name, group_idx in self._groups.items():
            gneeds_align = set()
            for name, map in getattr(self._data, mapattr).items():
                map = map[group_idx]
                if np.any(map == 0) or np.any(np.diff(map) != 1):
                    gneeds_align.add(name)
            self._needs_alignment[group_name] = gneeds_align

    def _get_groups(self, df: pd.DataFrame, group_by: str | None = None):
        if group_by is None:
            group_by = self._group_by
        return df.groupby(
            pd.Categorical(df[group_by]).rename_categories(lambda x: str(x))
            if group_by is not None
            else lambda x: self._dummy_group,
            observed=True,
        ).indices

    def _reindex(
        self, attr: str, axisattr: str, names: Mapping[str, NDArray[str]] | None = None
    ) -> pd.Index | slice | None:
        namesattr = f"{attr}_names"

        if names is not None and any(
            names[name].size != cnames.size or np.any(names[name] != cnames)
            for name, cnames in getattr(self, f"{axisattr}_names").items()
            if name in names
        ):
            selection = pd.Index(())
            for modname, mod in self._orig_data.mod.items():
                cnames = names.get(modname)
                if cnames is not None:
                    cnames = pd.Index(cnames)
                    if np.any(~cnames.isin(getattr(mod, namesattr))):
                        _logger.warning(
                            f"Not all names given for modality {modname} are present in the data. Restricting alignment to names present in the data."
                        )
                        cnames = cnames.intersection(getattr(mod, namesattr))
                else:
                    cnames = getattr(mod, namesattr)
                selection = selection.append(cnames)
        elif names is None:
            selection = slice(None)
        else:
            selection = None

        return selection

    @property
    def n_samples_total(self) -> int:
        return self._data.n_obs

    @property
    def n_features_total(self) -> int:
        return self._data.n_vars

    @property
    @abstractmethod
    def _subset_reorder(self):
        pass

    def _align_local_array_to_global_impl(
        self,
        arr: NDArray[T],
        name1: str | None,
        name2: str,
        subdata: MuData | None,
        align_to: Literal[0, 1],
        axis: int,
        fill_value: np.ScalarType,
        attr: str,
        param: str,
    ) -> NDArray[T]:
        if self._data.axis == 1 - align_to:
            return arr

        if subdata is None:
            if name1 is None:
                raise ValueError(f"Need either subdata or {param}, but both are None.")
            if name2 not in self._needs_alignment[name1]:
                return arr
            subdata = self._data[(self._groups[name1], slice(None))[self._subset_reorder]]

        idx = getattr(subdata, f"{attr}map")[name2]
        nnz = idx > 0

        outshape = [subdata.shape[self._data.axis]] + list(arr.shape[:axis]) + list(arr.shape[axis + 1 :])

        out = np.full(outshape, fill_value=fill_value, dtype=np.promote_types(type(fill_value), arr.dtype), order="C")
        out[nnz, ...] = np.moveaxis(arr, axis, 0)[idx[nnz] - 1, ...]
        return np.moveaxis(out, 0, axis)

    @MofaFlexDataset._axis_arg("align_to")
    def align_local_array_to_global(
        self,
        arr: NDArray[T],
        group_name: str,
        view_name: str,
        align_to: Literal[0, 1],
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ):
        return self._align_local_array_to_global(
            arr, view_name, group_name=group_name, align_to=align_to, axis=axis, fill_value=fill_value
        )

    @MofaFlexDataset._axis_arg("align_to")
    def align_global_array_to_local(
        self, arr: NDArray[T], group_name: str, view_name: str, align_to: Literal[0, 1], axis: int = 0
    ) -> NDArray[T]:
        if self._data.axis == align_to:
            idx = self.map_local_indices_to_global(slice(None), group_name, view_name, align_to)
            return np.take(arr, idx, axis=axis)
        else:
            return arr

    def _map_local_indices_to_global(
        self, idx: NDArray[int], name1: str, name2: str, align_to: Literal[0, 1], attr: str
    ) -> NDArray[int]:
        if self._data.axis == align_to:
            subdata = self._data[(self._groups[name1], slice(None))[self._subset_reorder]]
            map = getattr(subdata, f"{attr}map")[name2]
            mask = map > 0
            n = subdata.mod[name2].shape[align_to]
            _idx = np.empty(n, dtype=map.dtype)
            _idx[map[mask] - 1] = np.arange(n, dtype=map.dtype) + np.cumsum(~mask)[mask]

            return _idx[idx]
        else:
            return idx

    def _map_global_indices_to_local(
        self, idx: NDArray[int], name1: str, name2: str, align_to: Literal[0, 1], attr: str
    ) -> NDArray[int]:
        if self._data.axis == align_to:
            subdata = self._data[(self._groups[name1], slice(None))[self._subset_reorder]]
            return getattr(subdata, f"{attr}map")[name2][idx].astype(int) - 1
        else:
            return idx

    def _push_obs(self) -> MuData:
        # We don't want to duplicate MuData's push_obs logic, but at the same time
        # we don't want to modify the data object. So we create a temporary fake
        # MuData object with the same metadata, but no actual data
        fakeadatas = {
            modname: ad.AnnData(X=sparse.csr_array(mod.X.shape), obs=mod.obs, var=mod.var)
            for modname, mod in self._data.mod.items()
        }

        # need to pass obs in the constructor to make shape validation for obsmap work
        fakemudata = MuData(fakeadatas, obs=self._data.obs, obsmap=self._data.obsmap, axis=self._data.axis)
        # need to replace obs since the constructor runs update(), which breaks push_obs()
        fakemudata.obs = self._data.obs
        fakemudata.push_obs()
        return fakemudata

    def get_missing_obs(self) -> pd.DataFrame:
        dfs = []
        for _group_name, group_idx in self._groups.items():
            subdata = self._data[(group_idx, slice(None))[self._subset_reorder]]
            for modname, mod in subdata.mod.items():
                group_name, view_name = (_group_name, modname)[self._subset_reorder]
                if sparse.issparse(mod.X):
                    modmissing = mod.X.copy()
                    modmissing.data = np.isnan(modmissing.data)
                    modmissing = np.asarray(modmissing.sum(axis=1)).squeeze() == modmissing.shape[1]
                else:
                    modmissing = np.isnan(mod.X).all(axis=1)
                modmissing = self._align_local_array_to_global(modmissing, view_name, subdata, fill_value=True)
                dfs.append(
                    pd.DataFrame(
                        {
                            "view": view_name,
                            "group": group_name,
                            "obs_name": self.sample_names[group_name],
                            "missing": modmissing,
                        }
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
        outer_msg, inner_msg = ("group", "view")[dict_reorder][self._subset_reorder]

        covariates = defaultdict(dict)
        covar_dims = defaultdict(set)
        for group_name, group_idx in self._groups.items():
            if self._data.axis == axis and filter_names is not None and group_name not in filter_names:
                continue
            for modname in self._data.mod.keys():
                if self._data.axis != axis and filter_names is not None and modname not in filter_names:
                    continue
                subdata = self._data[(group_idx, self.get_names(1 - self._data.axis)[modname])[self._subset_reorder]]
                mod = subdata.mod[modname]
                outer_key, inner_key = (group_name, modname)[dict_reorder][self._subset_reorder]

                ckey = key.get(outer_key, None)
                cmkey = mkey.get(outer_key, None)

                if ckey is None and cmkey is None:
                    continue
                if ckey and cmkey:
                    raise ValueError(
                        f"Provide either key or mkey for {outer_msg} {outer_key}, got key='{ckey}', mkey='{cmkey}'."
                    )

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

    def _apply_to_axis_complement(
        self, name: str, func: ApplyToCallable[T], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data_for_apply()
        ret = {}
        for group_name, group_idx in self._groups.items():
            cret = func(
                data[(group_idx, slice(None))[self._subset_reorder]][name], group_name, **kwargs, **gkwargs[group_name]
            )
            ret[group_name] = apply_to_nested(cret, from_dask)
        return ret

    def _apply_to_axis(
        self, name: str, func: ApplyToCallable[T], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        data = self._data_for_apply()
        ret = {}
        data = data[(self._groups[name], slice(None))[self._subset_reorder]]
        for modname, mod in data.mod.items():
            cret = func(mod, modname, **kwargs, **gkwargs[modname])
            ret[modname] = apply_to_nested(cret, from_dask)
        return ret

    def _apply_by_axis_complement(
        self,
        func: ApplyCallable[T],
        names: Sequence[str],
        attr: str,
        gkwargs: Mapping[str, Mapping[str, Any]],
        **kwargs,
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
        attrmap = getattr(self._data, f"{attr}map")
        for modname in names:
            mod = data.mod[modname]
            groups = np.empty((mod.shape[self._data.axis],), dtype="O")
            for group, group_idx in self._groups.items():
                modidx = attrmap[modname][group_idx]
                modidx = modidx[modidx > 0] - 1
                groups[modidx] = group

            cret = func(mod, *(groups, modname)[self._subset_reorder], **kwargs, **gkwargs[modname])
            ret[modname] = apply_to_nested(cret, from_dask)
        return ret

    def _apply_by_axis(
        self,
        func: ApplyCallable[T],
        names: Sequence[str],
        attr: str,
        namesattr: str,
        gkwargs: Mapping[str, Mapping[str, Any]],
        **kwargs,
    ) -> dict[str, T]:
        data = self._data_for_apply()
        ret = {}
        for group_name in names:
            group_idx = self._groups[group_name]
            subdata = data[(group_idx, slice(None))[self._subset_reorder]]
            gdata = {}
            convert = False
            for modname, mod in subdata.mod.items():
                if mod.shape[subdata.axis] != subdata.shape[subdata.axis]:
                    convert = True
                gdata[modname] = mod
            if convert:
                for modname, mod in gdata.items():
                    mod = mod.copy()
                    mod.X = mod.X.astype(np.promote_types(mod.X.dtype, type(np.nan)))
                    gdata[modname] = mod
            gdata = ad.concat(
                gdata,
                axis=1 - subdata.axis,
                join="outer",
                label="____concat",
                merge="unique",
                uns_merge=None,
                fill_value=np.nan,
            )
            if (getattr(gdata, namesattr) != getattr(subdata, namesattr)).any():
                gdata = gdata[(getattr(subdata, namesattr), slice(None))[self._subset_reorder]]
            cret = func(
                gdata,
                *(group_name, getattr(gdata, attr)["____concat"].to_numpy())[self._subset_reorder],
                **kwargs,
                **gkwargs[group_name],
            )
            ret[group_name] = apply_to_nested(cret, from_dask)
        return ret


class MuDataAxis0Dataset(MuDataDataset):
    _dummy_group = "group_1"

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
        if feature_names is None and subset_var is not None:
            feature_names = {}
            if subset_var in mudata.var:
                for modname, modmap in mudata.varmap.items():
                    feature_names[modname] = mudata.mod[modname].var_names[mudata.var[subset_var][modmap.ravel() > 0]]
            else:
                for modname, mod in mudata.mod.items():
                    if subset_var in mod.var:
                        feature_names[modname] = mod.var_names[mod.var[subset_var]]
        super().__init__(
            mudata,
            layer=layer,
            group_by=group_by,
            preprocessor=preprocessor,
            cast_to=cast_to,
            subset_var=subset_var,
            sample_names=sample_names,
            feature_names=feature_names,
        )

    def reindex_samples(self, sample_names: Mapping[str, NDArray[str]] | None = None):
        selection = self._reindex_with_groups("obs", sample_names)
        if selection is not None:
            self._data = self._orig_data[selection, self._feature_selection]
            self._sample_selection = selection
        self._calc_groups("obs")

    def reindex_features(self, feature_names: Mapping[str, NDArray[str]] | None = None):
        selection = self._reindex("var", "feature", feature_names)
        if selection is not None:
            self._data = self._orig_data[self._sample_selection, selection]
            self._feature_selection = selection

    @staticmethod
    def _accepts_input(data):
        return isinstance(data, MuData) and data.axis == 0

    @property
    def _subset_reorder(self):
        return slice(None)

    @property
    def n_features(self) -> dict[str, int]:
        return {modname: mod.n_vars for modname, mod in self._data.mod.items()}

    @property
    def n_samples(self) -> dict[str, int]:
        return {groupname: len(groupidx) for groupname, groupidx in self._groups.items()}

    @property
    def view_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._data.mod.keys()))

    @property
    def group_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._groups.keys()))

    @property
    def sample_names(self) -> dict[str, NDArray[str]]:
        return {groupname: self._data.obs_names[groupidx].to_numpy() for groupname, groupidx in self._groups.items()}

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
                group[modname] = arr
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
        align_to: Literal[0, 1] = 0,
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ) -> NDArray[T]:
        return self._align_local_array_to_global_impl(
            arr, group_name, view_name, subdata, align_to, axis, fill_value, "obs", "group_name"
        )

    @MofaFlexDataset._axis_arg("align_to")
    def map_local_indices_to_global(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal[0, 1]
    ) -> NDArray[int]:
        return self._map_local_indices_to_global(idx, group_name, view_name, align_to, "obs")

    @MofaFlexDataset._axis_arg("align_to")
    def map_global_indices_to_local(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal[0, 1]
    ) -> NDArray[int]:
        return self._map_global_indices_to_local(idx, group_name, view_name, align_to, "obs")

    def get_obs(self) -> dict[str, pd.DataFrame]:
        fakemudata = self._push_obs()
        return {
            group_name: {
                modname: mod.obs.reindex(self._data[group_idx, :].obs_names, fill_value=pd.NA).apply(
                    lambda x: x.astype("string") if x.dtype == "O" else x, axis=1
                )
                for modname, mod in fakemudata.mod.items()
            }
            for group_name, group_idx in self._groups.items()
        }

    def _apply_to_view(
        self, view_name: str, func: ApplyToCallable[T], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        return self._apply_to_axis_complement(view_name, func, gkwargs, **kwargs)

    def _apply_to_group(
        self, group_name: str, func: ApplyToCallable[T], vkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        return self._apply_to_axis(group_name, func, vkwargs, **kwargs)

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
        return self._apply_by_axis_complement(func, view_names, "obs", vkwargs, **kwargs)

    def _apply_by_group(
        self, func: ApplyCallable[T], group_names: Sequence[str], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        return self._apply_by_axis(func, group_names, "var", "obs_names", gkwargs, **kwargs)


class MuDataAxis1Dataset(MuDataDataset):
    _dummy_group = "view_1"

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
        if feature_names is None and subset_var is not None:
            feature_names = {}
            groups = self._get_groups(mudata.var, group_by)
            if subset_var in mudata.var:
                for view_name, view_idx in groups.items():
                    names = mudata.var[subset_var].iloc[view_idx]
                    feature_names[view_name] = names.index[names]
            else:
                for modname, mod in mudata.mod.items():
                    modmap = mudata.varmap[modname].ravel()
                    modmask = modmap > 0
                    if subset_var in mod.var:
                        for view_name, view_idx in groups.items():
                            cmask = np.zeros_like(modmask)
                            cmask[view_idx] = 1
                            cmask &= modmask
                            cnames = mod.var[subset_var].iloc[modmap[cmask] - 1]
                            cnames = cnames.index[cnames]
                            if view_name not in feature_names:
                                feature_names[view_name] = cnames
                            else:
                                feature_names[view_name] = feature_names[view_name].intersection(cnames)
        else:
            groups = None
        super().__init__(
            mudata,
            layer=layer,
            group_by=group_by,
            preprocessor=preprocessor,
            cast_to=cast_to,
            subset_var=subset_var,
            sample_names=sample_names,
            feature_names=feature_names,
            groups=groups,
        )

    def reindex_samples(self, sample_names: Mapping[str, NDArray[str]] | None = None):
        selection = self._reindex("obs", "sample", sample_names)
        if selection is not None:
            self._data = self._orig_data[selection, self._feature_selection]
            self._sample_selection = selection

    def reindex_features(self, feature_names: Mapping[str, NDArray[str]] | None = None):
        selection = self._reindex_with_groups("var", feature_names)
        if selection is not None:
            self._data = self._orig_data[self._sample_selection, selection]
            self._feature_selection = selection
        self._calc_groups("var")
        self._nonmissing_var = {}
        for view_name, view_idx in self._groups.items():
            vnonmissing_var = {}
            for modname in self._needs_alignment[view_name]:
                vnonmissing_var[modname] = np.nonzero(self._data.varmap[modname][view_idx] > 0)[0]
            self._nonmissing_var[view_name] = vnonmissing_var

    @staticmethod
    def _accepts_input(data):
        return isinstance(data, MuData) and data.axis == 1

    @property
    def _subset_reorder(self):
        return slice(None, None, -1)

    @property
    def n_samples(self) -> dict[str, int]:
        return {modname: mod.n_obs for modname, mod in self._data.mod.items()}

    @property
    def n_features(self) -> dict[str, int]:
        return {viewname: len(groupidx) for viewname, groupidx in self._groups.items()}

    @property
    def group_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._data.mod.keys()))

    @property
    def view_names(self) -> NDArray[str]:
        return np.asarray(tuple(self._groups.keys()))

    @property
    def feature_names(self) -> dict[str, NDArray[str]]:
        return {viewname: self._data.var_names[groupidx].to_numpy() for viewname, groupidx in self._groups.items()}

    @property
    def sample_names(self) -> dict[str, NDArray[str]]:
        return {groupname: mod.obs_names.to_numpy() for groupname, mod in self._data.mod.items()}

    def __getitems__(self, idx: Mapping[str, int | Sequence[int]]) -> dict[str, dict]:
        data = {}
        nonmissing_obs = {}
        nonmissing_var = {}
        for group_name, group_idx in idx.items():
            group = {}
            gnonmissing_obs = {}
            gnonmissing_var = {}
            for view_name, view_idx in self._groups.items():
                cnonmissing_var = self._nonmissing_var[view_name].get(group_name, slice(None))
                arr, gnonmissing_obs[view_name], gnonmissing_var[view_name] = self.preprocessor(
                    self._data[:, view_idx][group_name].X[group_idx, :],
                    slice(None),
                    cnonmissing_var,
                    group_name,
                    view_name,
                )
                if self.cast_to is not None:
                    arr = arr.astype(self.cast_to, copy=False)
                group[view_name] = arr
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
        align_to: Literal[0, 1] = 0,
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ) -> NDArray[T]:
        return self._align_local_array_to_global_impl(
            arr, view_name, group_name, subdata, align_to, axis, fill_value, "var", "view_name"
        )

    @MofaFlexDataset._axis_arg("align_to")
    def map_local_indices_to_global(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal[0, 1]
    ) -> NDArray[int]:
        return self._map_local_indices_to_global(idx, view_name, group_name, align_to, "var")

    @MofaFlexDataset._axis_arg("align_to")
    def map_global_indices_to_local(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal[0, 1]
    ) -> NDArray[int]:
        return self._map_global_indices_to_local(idx, view_name, group_name, align_to, "var")

    def get_obs(self) -> dict[str, pd.DataFrame]:
        fakemudata = self._push_obs()
        return {modname: dict.fromkeys(self._groups.keys(), mod.obs) for modname, mod in fakemudata.mod.items()}

    def _apply_to_view(
        self, view_name: str, func: ApplyToCallable[T], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        return self._apply_to_axis(view_name, func, gkwargs, **kwargs)

    def _apply_to_group(
        self, group_name: str, func: ApplyToCallable[T], vkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        return self._apply_to_axis_complement(group_name, func, vkwargs, **kwargs)

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
        for view_name in view_names:
            view_idx = self._groups[view_name]
            subdata = data[:, view_idx]
            for modname in group_names:
                ccret = func(subdata.mod[modname], modname, view_name, **kwargs, **gvkwargs[modname][view_name])
                ret.get(modname, {})[view_name] = ccret
        return ret

    def _apply_by_view(
        self, func: ApplyCallable[T], view_names: Sequence[str], vkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        return self._apply_by_axis(func, view_names, "obs", "var_names", vkwargs, **kwargs)

    def _apply_by_group(
        self, func: ApplyCallable[T], group_names: Sequence[str], gkwargs: Mapping[str, Mapping[str, Any]], **kwargs
    ) -> dict[str, T]:
        return self._apply_by_axis_complement(func, group_names, "var", gkwargs, **kwargs)
