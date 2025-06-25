from collections import namedtuple
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import (
    coo_array,
    coo_matrix,
    csc_array,
    csc_matrix,
    csr_array,
    csr_matrix,
    issparse,
    lil_array,
    sparray,
    spmatrix,
)
from torch.utils.data import BatchSampler, SequentialSampler

from .datasets import MofaFlexDataset

PossiblySparseArray: TypeAlias = NDArray | spmatrix | sparray

MeanStd = namedtuple("MeanStd", ["mean", "std"])
ShapeRate = namedtuple("ShapeRate", ["shape", "rate"])


def sample_all_data_as_one_batch(data: MofaFlexDataset) -> dict[str, list[int]]:
    return {
        k: next(
            iter(BatchSampler(SequentialSampler(range(nsamples)), batch_size=data.n_samples_total, drop_last=False))
        )
        for k, nsamples in data.n_samples.items()
    }


def mean(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if issparse(arr):
        mean = np.asarray(arr.mean(axis=axis))
        if not keepdims and axis is not None and mean.ndim == arr.ndim:
            mean = mean.squeeze(axis)
        elif keepdims and mean.ndim < arr.ndim:
            if axis is None:
                mean = np.expand_dims(mean, tuple(range(arr.ndim)))
            else:
                mean = np.expand_dims(mean, axis=axis)

    else:
        mean = arr.mean(axis=axis, keepdims=keepdims)
    return mean


# TODO: use numba for this?
def _nanmean_cs_aligned(arr: csr_array | csr_matrix | csc_array | csc_matrix):
    axis = 1 if isinstance(arr, csr_array | csr_matrix) else 0
    out = np.empty(arr.shape[1 - axis], dtype=np.float64 if np.issubdtype(arr.dtype, np.integer) else arr.dtype)
    for r in range(out.size):
        data = arr.data[arr.indptr[r] : arr.indptr[r + 1]]
        mask = np.isnan(data)
        out[r] = data[~mask].sum() / (arr.shape[axis] - mask.sum())
    return out


# TODO: use numba for this?
def _nanmean_cs_nonaligned(arr: csr_array | csr_matrix | csc_array | csc_matrix):
    axis = 0 if isinstance(arr, csr_array | csr_matrix) else 1
    out = np.zeros(arr.shape[1 - axis], dtype=np.float64 if np.issubdtype(arr.dtype, np.integer) else arr.dtype)
    n = np.full(out.size, fill_value=arr.shape[axis], dtype=np.uint32)
    for r in range(arr.shape[axis]):
        idx = arr.indices[arr.indptr[r] : arr.indptr[r + 1]]
        data = arr.data[arr.indptr[r] : arr.indptr[r + 1]]
        mask = np.isnan(data)
        out[idx[~mask]] += data[~mask]
        n[idx[mask]] -= 1
    out /= n
    return out


def nanmean(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if issparse(arr):
        if axis is None:
            mean = np.nansum(arr.data) / (np.prod(arr.shape) - np.sum(np.isnan(arr.data)))
            if keepdims:
                mean = mean[None, None]
        else:
            if (
                axis == 0
                and isinstance(arr, csr_array | csr_matrix)
                or axis == 1
                and isinstance(arr, csc_array | csc_matrix)
            ):
                mean = _nanmean_cs_nonaligned(arr)
            elif (
                axis == 1
                and isinstance(arr, csr_array | csr_matrix)
                or axis == 0
                and isinstance(arr, csc_array | csc_matrix)
            ):
                mean = _nanmean_cs_aligned(arr)
            else:
                raise NotImplementedError(f"Unsupported sparse matrix type {type(arr)}.")
            if keepdims:
                mean = np.expand_dims(mean, axis)
    else:
        mean = np.nanmean(arr, axis=axis, keepdims=keepdims)
    return mean


def var(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if issparse(arr):
        _mean = mean(arr, axis=axis, keepdims=True)
        var = (np.asarray(arr - _mean) ** 2).mean(axis=axis, keepdims=keepdims)
    else:
        var = arr.var(axis=axis, keepdims=keepdims)
    return var


def nanvar(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    if issparse(arr):
        _mean = nanmean(arr, axis=axis, keepdims=True)
        var = np.nanmean(np.asarray(arr - _mean) ** 2, axis=axis, keepdims=keepdims)
    else:
        var = np.nanvar(arr, axis=axis, keepdims=keepdims)
    return var


def min(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    return _minmax(arr, method="min", axis=axis, keepdims=keepdims)


def max(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    return _minmax(arr, method="max", axis=axis, keepdims=keepdims)


def nanmin(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    return _minmax(arr, method="nanmin", axis=axis, keepdims=keepdims)


def nanmax(arr: PossiblySparseArray, axis: int | None = None, keepdims=False):
    return _minmax(arr, method="nanmax", axis=axis, keepdims=keepdims)


def wherenan(arr: PossiblySparseArray):
    if not issparse(arr):
        return np.nonzero(np.isnan(arr))
    else:
        nanidx = np.nonzero(np.isnan(arr.data))[0]
        need_sort = False
        if isinstance(arr, coo_array | coo_matrix):
            rowidx, colidx = arr.data[:, 0], arr.data[:, 1]
            need_sort = True
        elif isinstance(arr, csr_array | csr_matrix | csc_array | csc_matrix):
            colidx = arr.indices[nanidx]
            rowidx = np.searchsorted(arr.indptr, nanidx, side="right") - 1
            if isinstance(arr, csc_array | csc_matrix):
                colidx, rowidx = rowidx, colidx
                need_sort = True
        else:
            raise NotImplementedError(f"Unsupported sparse matrix type {type(arr)}.")

        if need_sort:  # be compatible with np.nonzero, which returns sorted results
            order = np.argsort(rowidx, stable=True)
            rowidx, colidx = rowidx[order], colidx[order]
        return rowidx, colidx


def _minmax(
    arr: PossiblySparseArray, method: Literal["min", "max", "nanmin", "nanmax"], axis: int | None = None, keepdims=False
):
    if np.prod(arr.shape) == 0:
        return arr.reshape((0,) * arr.ndim)
    if hasattr(arr, method):
        res = getattr(arr, method)(axis=axis)
    else:
        res = getattr(np, method)(arr, axis=axis)
    if issparse(res):
        res = res.toarray()
    if keepdims and res.ndim < arr.ndim:
        res = np.expand_dims(res, axis if axis is not None else tuple(range(arr.ndim)))
    elif not keepdims and res.ndim == arr.ndim:
        res = res.squeeze(axis)
    return res


def impute(
    data: AnnData,
    group_name,
    view_name,
    factors,
    weights,
    sample_names,
    feature_names,
    likelihood,
    missingonly,
    preprocessor,
):
    havemissing = data.n_obs < factors.shape[0] or data.n_vars < weights.shape[1]
    if issparse(data.X):
        have_missing_cells = np.isnan(data.X.data).sum() > 0
    else:
        have_missing_cells = np.isnan(data.X).sum() > 0
    havemissing |= have_missing_cells

    if missingonly and not havemissing:
        return data
    elif not missingonly:
        imputation = likelihood.transform_prediction(factors @ weights, preprocessor.sample_means)
    else:
        missing_obs = align_local_array_to_global(  # noqa F821
            np.broadcast_to(False, (data.n_obs,)), group_name, view_name, fill_value=True, align_to="samples"
        )
        missing_var = align_local_array_to_global(  # noqa F821
            np.broadcast_to(False, (data.n_vars)), group_name, view_name, fill_value=True, align_to="features"
        )

        preprocessed = preprocessor(data.X, slice(None), slice(None), group_name, view_name)[0]
        if issparse(preprocessed):
            imputation = lil_array((factors.shape[0], weights.shape[1]))
        else:
            imputation = np.empty((sample_names.size, feature_names.size), dtype=data.X.dtype)

        obsidx = map_local_indices_to_global(np.arange(data.n_obs), group_name, view_name, align_to="samples")  # noqa F821
        varidx = map_local_indices_to_global(np.arange(data.n_vars), group_name, view_name, align_to="features")  # noqa F821
        imputation[np.ix_(obsidx, varidx)] = preprocessed

        if issparse(data.X):
            for row in np.nonzero(missing_obs)[0]:
                imputation[row, :] = likelihood.transform_prediction(
                    factors[row, :] @ weights, preprocessor.sample_means
                )
            imputation = imputation.T  # slow column slicing for lil arrays
            for col in np.nonzero(missing_var)[0]:
                imputation[col, :] = likelihood.transform_prediction(
                    factors @ weights[:, col], preprocessor.sample_means
                ).T
            imputation = imputation.T
        else:
            imputation[missing_obs, :] = likelihood.transform_prediction(
                factors[missing_obs, :] @ weights, preprocessor.sample_means
            )
            imputation[:, missing_var] = likelihood.transform_prediction(
                factors @ weights[:, missing_var], preprocessor.sample_means
            )

        if have_missing_cells:
            nanobs, nanvar = wherenan(data.X)
            nanobs, nanvar = obsidx[nanobs], varidx[nanvar]
            imputation[nanobs, nanvar] = likelihood.transform_prediction(
                (factors[nanobs, :] * weights[:, nanvar].T).sum(axis=1), preprocessor.sample_means
            )

        if issparse(data.X):
            imputation = imputation.tocsr()

    return AnnData(X=imputation, obs=pd.DataFrame(index=sample_names), var=pd.DataFrame(index=feature_names))
