from collections import namedtuple
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import MISSING, dataclass, fields
from inspect import isabstract, signature
from io import BytesIO
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from numpy.typing import NDArray
from pyro.nn import PyroModule
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

if TYPE_CHECKING:
    from .datasets import MofaFlexDataset

PossiblySparseArray: TypeAlias = NDArray | spmatrix | sparray

MeanStd = namedtuple("MeanStd", ["mean", "std"])
ShapeRate = namedtuple("ShapeRate", ["shape", "rate"])

PyroParameterDict = PyroModule[torch.nn.ParameterDict]
PyroModuleDict = PyroModule[torch.nn.ModuleDict]


def checked_baseclass(
    required_init_args: Sequence[str] | str = (),
    required_init_kwargs: Sequence[str] | str = (),
    required_init_kkwargs: bool = False,
    required_attributes: Sequence[str] | str = (),
    registry: Literal["set", "dict", None] = None,
):
    if isinstance(required_init_args, str):
        required_init_args = (required_init_args,)
    if isinstance(required_init_kwargs, str):
        required_init_kwargs = (required_init_kwargs,)
    if isinstance(required_attributes, str):
        required_attributes = (required_attributes,)

    def decorate(cls: type):
        subinitcls = cls.__dict__.get("__init_subclass__", None)
        if subinitcls is not None:
            subinitcls = subinitcls.__get__(cls, cls)

        def init_subclass(subcls, **kwargs):
            super(cls).__init_subclass__(**kwargs)
            if subinitcls is not None:
                subinitcls(**kwargs)
            if not isabstract(subcls) and subcls.__name__[0] != "_":
                init_sig = signature(subcls.__init__)
                for i, (required_arg, param) in enumerate(
                    zip(required_init_args, islice(init_sig.parameters.values(), 1, None), strict=False)
                ):
                    if required_arg != param.name:
                        raise TypeError(
                            f"Constructor of class {subcls} is missing the {required_arg} argument at position {i + 1}."
                        )
                for required_arg in required_init_kwargs:
                    if required_arg not in init_sig.parameters:
                        raise TypeError(f"Constructor of class {subcls} is missing the {required_arg} argument.")
                if required_init_kkwargs and "kwargs" not in init_sig.parameters:
                    raise TypeError(f"Constructor of class {subcls} is missing the {kwargs} argument.")

                for required_attr in required_attributes:
                    if not hasattr(subcls, required_attr):
                        raise TypeError(f"Class {subcls} is missing the {required_attr} attribute.")

                if registry == "set":
                    cls._registry.add(subcls)
                elif registry == "dict":
                    cls._registry[subcls.__name__] = subcls

                    subinit = subcls.__dict__.get("__init__", None)

                    def init(self, *args, **kwargs):
                        if len(args) > len(init_sig.parameters) and subcls is not cls and args[0] == subcls.__name__:
                            args = args[1:]
                        if subinit is not None:
                            subinit(self, *args, **kwargs)
                        else:
                            super(subcls, self).__init__(*args, **kwargs)

                    if subinit is not None:
                        init.__signature__ = signature(subinit)
                        init.__annotations__ = subinit.__annotations__
                        init.__doc__ = subinit.__doc__

                    subcls.__init__ = init

        cls.__init_subclass__ = classmethod(init_subclass)

        if registry == "dict":

            def new(ccls, *args, **kwargs):
                if ccls is not cls or len(args) == 0 or not isinstance(args[0], str):
                    return super(cls, cls).__new__(ccls)
                try:
                    subclsname = args[0]
                    subcls = ccls._registry[ccls.name]
                    return subcls.__new__(subcls, *args[1:], **kwargs)
                except KeyError as e:
                    raise NotImplementedError(f"Uknown {cls.__name__.lower()} {subclsname}.") from e

            cls._registry = {}
            cls.__new__ = new
        elif registry == "set":
            cls._registry = set()

        return cls

    return decorate


class SaveStateMixin:
    def save(self) -> dict[str, Any]:
        """Called by the model to save its state to disk.

        If a subclass has a class attribute `_state_attrs`, which is a sequence of strings, each element of this list is used
        as the name of an instance variable to be saved to disk. Similarly, if a subclass has a class attribute `_state_attrs_meanstd`,
        which is a sequence of strings, each element of this list is assumed to be an instance variable of type `MeanStd` to be saved
        to disk. Subclasses must not reimplement this method. If custom behavior is desired, reimplement `_save` instead.
        """
        state = {}
        state_meanstd = {}

        cls = self.__class__
        while cls is not None:
            with suppress(AttributeError):
                for attrname in cls._state_attrs:
                    if isinstance(attr := getattr(self, attrname), MeanStd):
                        state_meanstd[attrname] = attr._asdict()
                    else:
                        state[attrname] = attr
            for base in cls.__bases__:
                if issubclass(base, __class__):
                    cls = base
                    break
            else:
                cls = None

        state.update(self._save())
        return {"class": self.__class__.__name__, "state": state, "state_meanstd": state_meanstd}

    def _save(self) -> dict[str, Any]:
        """Hook to save a prior's state to disk."""
        return {}

    @classmethod
    def load(
        cls, state: dict[str, Any], n_samples: dict[str, int], n_features: dict[str, int], map_location=None, **kwargs
    ):
        """Called by the model to restore its state from disk.

        If a subclass has a class attribute `state_attrs`, which is a sequence of strings, each element of this list is used
        as the name of an instance variable to be restored. Similarly, if a subclass has a class attribute `_state_attrs_meanstd`,
        which is a sequence of strings, each element of this list is assumed to be an instance variable of type `MeanStd` to be
        restored.Subclasses must not reimplement this method. If custom behavior is desired, reimplement `_load` instead.

        Args:
            state: The saved state.
            n_samples: The number of samples in each group.
            n_features: The number of features in each group.
            map_location: A device to map any potential PyTorch state to.
            **kwargs: Additional arguments to `_load`.
        """
        try:
            subcls = cls._registry[state["class"]]
            obj = subcls.__new__(subcls)
        except (KeyError, AttributeError):
            obj = __class__.__new__(cls)
        if isinstance(obj, PyroModule):
            PyroModule.__init__(obj)
        elif isinstance(obj, torch.nn.Module):
            torch.nn.Module.__init__(obj)

        for attrname, attr in state["state_meanstd"].items():
            setattr(obj, attrname, MeanStd(**attr))
        substate = state["state"]
        for attrname, attr in substate.items():
            setattr(obj, attrname, attr)
        obj._load(substate, n_samples, n_features, map_location=map_location, **kwargs)
        return obj

    def _load(
        self,
        state: dict[str, Any],
        n_samples: dict[str, int],
        n_features: dict[str, int],
        *,
        map_location=None,
        **kwargs,
    ):
        """Hook to load a prior's state from disk.

        Args:
            state: The saved state.
            n_samples: The number of samples in each group.
            n_features: The number of features in each group.
            map_location: A device to map any potential PyTorch state to.
            **kwargs: Additional, class-specific, arguments.
        """
        pass


@dataclass(kw_only=True)
class Options:
    def __or__(self, other):
        if self.__class__ is not other.__class__:
            raise TypeError("Can only merge objects of the same type")

        kwargs = self.asdict()
        for f in fields(other):
            val = getattr(other, f.name)
            if (
                f.default is not MISSING
                and val != f.default
                or f.default_factory is not MISSING
                and val != f.default_factory()
            ):
                kwargs[f.name] = val
        return self.__class__(**kwargs)

    def __ior__(self, other):
        if self.__class__ is not other.__class__:
            raise TypeError("Can only merge objects of the same type")

        for f in fields(other):
            val = getattr(other, f.name)
            if (
                f.default is not MISSING
                and val != f.default
                or f.default_factory is not MISSING
                and val != f.default_factory()
            ):
                setattr(self, f.name, val)
        return self

    def asdict(self):
        # avoid the deepcopy done by dataclasses.asdict
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def __post_init__(self):
        # after an HDF5 roundtrip, these are numpy scalars, which PyTorch doesn't handle well'
        for f in fields(self):
            if f.type in (float, int, bool):
                setattr(self, f.name, f.type(getattr(self, f.name)))


def pickle_torch_state(state: dict) -> NDArray[np.uint8]:
    pkl = BytesIO()
    torch.save(state, pkl)
    return np.frombuffer(pkl.getbuffer(), dtype=np.uint8)


def unpickle_torch_state(state: NDArray[np.uint8], map_location=None):
    pkl = BytesIO(state.tobytes())
    return torch.load(pkl, map_location=map_location, weights_only=True)


def sample_all_data_as_one_batch(data: "MofaFlexDataset") -> dict[str, list[int]]:
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
    havemissing = data.n_obs < factors.shape[0] or data.n_vars < weights.shape[0]
    if issparse(data.X):
        have_missing_cells = np.isnan(data.X.data).sum() > 0
    else:
        have_missing_cells = np.isnan(data.X).sum() > 0
    havemissing |= have_missing_cells

    if missingonly and not havemissing:
        return data

    if not missingonly:
        imputation = likelihood.transform_prediction(factors @ weights.T, preprocessor.sample_means)
    else:
        missing_obs = align_local_array_to_global(  # noqa F821
            np.broadcast_to(False, (data.n_obs,)), group_name, view_name, fill_value=True, align_to="samples"
        )
        missing_var = align_local_array_to_global(  # noqa F821
            np.broadcast_to(False, (data.n_vars)), group_name, view_name, fill_value=True, align_to="features"
        )

        preprocessed = preprocessor(data.X, slice(None), slice(None), group_name, view_name)[0]
        if issparse(preprocessed):
            imputation = lil_array((factors.shape[0], weights.shape[0]))
        else:
            imputation = np.empty((sample_names.size, feature_names.size), dtype=data.X.dtype)

        obsidx = map_local_indices_to_global(np.arange(data.n_obs), group_name, view_name, align_to="samples")  # noqa F821
        varidx = map_local_indices_to_global(np.arange(data.n_vars), group_name, view_name, align_to="features")  # noqa F821
        imputation[np.ix_(obsidx, varidx)] = preprocessed

        if issparse(data.X):
            for row in np.nonzero(missing_obs)[0]:
                imputation[row, :] = likelihood.transform_prediction(
                    factors[row, :] @ weights.T, preprocessor.sample_means
                )
            imputation = imputation.T  # slow column slicing for lil arrays
            for col in np.nonzero(missing_var)[0]:
                imputation[col, :] = likelihood.transform_prediction(
                    factors @ weights[col, :].T, preprocessor.sample_means
                ).T
            imputation = imputation.T
        else:
            imputation[missing_obs, :] = likelihood.transform_prediction(
                factors[missing_obs, :] @ weights.T, preprocessor.sample_means
            )
            imputation[:, missing_var] = likelihood.transform_prediction(
                factors @ weights[missing_var, :].T, preprocessor.sample_means
            )

        if have_missing_cells:
            nanobs, nanvar = wherenan(data.X)
            nanobs, nanvar = np.atleast_1d(obsidx[nanobs]), np.atleast_1d(varidx[nanvar])
            imputation[nanobs, nanvar] = likelihood.transform_prediction(
                (factors[nanobs, :] * weights[nanvar, :]).sum(axis=1), preprocessor.sample_means
            )

        if issparse(data.X):
            imputation = imputation.tocsr()

    return AnnData(X=imputation, obs=pd.DataFrame(index=sample_names), var=pd.DataFrame(index=feature_names))
