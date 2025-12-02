import logging
from collections import namedtuple
from collections.abc import Callable, Mapping, Sequence, Set
from contextlib import suppress
from importlib.util import find_spec
from typing import Any, Union

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
    sparray,
    spmatrix,
)

from ..settings import settings

AlignmentMap = namedtuple("AlignmentMap", ["d2g", "g2d"])

_logger = logging.getLogger(__name__)


def have_dask():
    return find_spec("dask") is not None


_warned_sparse = False


def array_to_dask(arr: NDArray | spmatrix | sparray | pd.DataFrame):
    import dask.array as da
    import dask.dataframe as dd

    if isinstance(arr, pd.DataFrame):
        return dd.from_pandas(arr, sort=False)

    elemsize = arr.dtype.itemsize

    chunksize = settings.dask_chunksize_mb * 1024 * 1024
    if issparse(arr):
        # https://github.com/pydata/sparse/issues/860
        # https://github.com/dask/dask/issues/11880
        global _warned_sparse
        if not _warned_sparse:
            _logger.warning(
                "Sparse arrays are currently not supported by Dask. Dask will not be used"
                " and data arrays may be copied, resulting in high memory usage."
            )
            _warned_sparse = True
        return arr

        import os

        os.environ["SPARSE_AUTO_DENSIFY"] = "1"  # https://github.com/pydata/sparse/issues/842
        import sparse

        if isinstance(arr, csr_array | csr_matrix):
            arr.sort_indices()
            arr = sparse.GCXS((arr.data, arr.indices, arr.indptr), shape=arr.shape, compressed_axes=(0,))

            chunks = (chunksize // (arr.shape[1] * elemsize), -1)
        elif isinstance(arr, csc_array | csc_matrix):
            arr.sort_indices()
            arr = sparse.GCXS((arr.data, arr.indices, arr.indptr), shape=arr.shape, compressed_axes=(1,))

            chunks = (-1, chunksize // (arr.shape[0] * elemsize))
        elif isinstance(arr, coo_array):
            arr = sparse.COO(arr.coords, arr.data, shape=arr.shape)
            chunks = (-1, -1)
        elif isinstance(arr, coo_matrix):
            arr = sparse.COO(np.stack((arr.row, arr.col), axis=0), arr.data, shape=arr.shape)
            chunks = (-1, -1)
        else:
            arr = sparse.asarray(arr, format="csr")
            chunks = (chunksize // (arr.shape[1] * elemsize), -1)
    else:
        chunks = (-1, -1)
    return da.from_array(arr, chunks=chunks)


def from_dask(arr, convert_coo=True):
    if type(arr).__module__.startswith("dask."):
        arr = arr.compute()
    if type(arr).__module__.startswith("sparse."):
        import os

        os.environ["SPARSE_AUTO_DENSIFY"] = "1"  # https://github.com/pydata/sparse/issues/842

        if arr.ndim == 2:
            arr = arr.to_scipy_sparse()
            if convert_coo and isinstance(arr, coo_array | coo_matrix):
                arr = arr.tocsr()
        else:
            arr = arr.todense()
    return arr


def apply_to_nested(data, func: Callable[[Any], Any]):
    if isinstance(data, Mapping):
        return type(data)({k: apply_to_nested(v, func) for k, v in data.items()})
    elif isinstance(data, tuple):
        args = (apply_to_nested(v, func) for v in data)
        if hasattr(data, "_fields"):  # namedtuple
            return type(data)(*args)
        else:
            return type(data)(args)
    elif isinstance(data, Sequence | Set) and not isinstance(data, str | bytes):
        return type(data)(apply_to_nested(v, func) for v in data)
    else:
        return func(data)


def anndata_to_dask(adata: AnnData):
    dask_adata = AnnData(
        X=array_to_dask(adata.X), var=adata.var, obs=adata.obs
    )  # AnnData does not support Dask DataFrames for var and obs
    return dask_adata


_warned_dask = False


def warn_dask(logger: logging.Logger | None = None):
    global _warned_dask
    if _warned_dask:
        return
    if logger is None:
        logger = _logger
    logger.warning("Could not import dask. Data arrays may be copied, resulting in high memory usage.")
    _warned_dask = True


def select_anndata_layer(adata: AnnData, layer: str | None = None):
    if layer is None:
        return adata
    else:
        return AnnData(
            X=adata.layers[layer],
            obs=adata.obs,
            var=adata.var,
            obsm=adata.obsm,
            varm=adata.varm,
            obsp=adata.obsp,
            varp=adata.varp,
            uns=adata.uns,
        )


def align_dataframe(
    df,
    new_index,
    fill_value: Callable[[np.dtype | pd.api.extensions.ExtensionDtype], Union[*np.ScalarType]] = lambda _: pd.NA,
):
    df = df.convert_dtypes()
    new_cols = {}
    for colname in df.columns:
        col = df[colname]
        new_cols[colname] = col.reindex(new_index, fill_value=fill_value(col.dtype))
    for colname, col in new_cols.items():
        with suppress(ValueError):
            match col.dtype:
                case pd.BooleanDtype():
                    col = col.astype(bool)
                case (
                    pd.core.arrays.integer.IntegerDtype(type=dtype) | pd.core.arrays.floating.FloatingDtype(type=dtype)
                ):
                    col = col.astype(dtype)
        new_cols[colname] = col
    return pd.DataFrame(new_cols)
