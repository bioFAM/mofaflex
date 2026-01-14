import logging
import os
from abc import ABC
from collections import namedtuple
from collections.abc import Iterable, Mapping, Sequence
from contextlib import suppress
from inspect import isabstract, signature
from io import BytesIO
from itertools import islice
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import torch
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
    sparray,
    spmatrix,
)
from torch.utils.data import BatchSampler, SequentialSampler

from .settings import settings

if TYPE_CHECKING:
    from .datasets import MofaFlexDataset

_logger = logging.getLogger(__name__)

PossiblySparseArray: TypeAlias = NDArray | spmatrix | sparray

MeanStd = namedtuple("MeanStd", ["mean", "std"])
ShapeRate = namedtuple("ShapeRate", ["shape", "rate"])

PyroParameterDict = PyroModule[torch.nn.ParameterDict]
PyroModuleDict = PyroModule[torch.nn.ModuleDict]


# https://stackoverflow.com/a/61350480
class _PyroMeta(type(ABC), type(PyroModule)):
    pass


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
                            f"Constructor of class {subcls} is missing the '{required_arg}' argument at position {i + 1}."
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
                        if (
                            len(args) > 0
                            and subcls is not cls
                            and isinstance(args[0], str)
                            and args[0] == subcls.__name__
                        ):
                            args = args[1:]
                        if subinit is not None:
                            subinit(self, *args, **kwargs)
                        else:
                            super(subcls, self).__init__(*args, **kwargs)

                    if subinit is None:
                        subinit = subcls.__init__
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
                    subcls = ccls._registry[subclsname]
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
    @classmethod
    def _get_state_attrs(cls) -> Iterable[str]:
        while cls is not None:
            with suppress(AttributeError):
                if isinstance(attrs := cls._state_attrs, str):
                    yield attrs
                else:
                    yield from attrs
            for base in cls.__bases__:
                if issubclass(base, __class__):
                    cls = base
                    break
            else:
                cls = None

    def save(self) -> dict[str, Any]:
        """Called by the model to save its state to disk.

        If a subclass has a class attribute `_state_attrs`, which is a sequence of strings, each element of this list is used
        as the name of an instance variable to be saved to disk. Similarly, if a subclass has a class attribute `_state_attrs_meanstd`,
        which is a sequence of strings, each element of this list is assumed to be an instance variable of type `MeanStd` to be saved
        to disk. Subclasses must not reimplement this method. If custom behavior is desired, reimplement `_save` instead.
        """
        state = {}
        state_meanstd = {}

        for attrname in self._get_state_attrs():
            with suppress(AttributeError):
                if isinstance(attr := getattr(self, attrname), MeanStd):
                    state_meanstd[attrname] = attr._asdict()
                else:
                    state[attrname] = attr

        state.update(self._save())
        return {"class": self.__class__.__name__, "state": state, "state_meanstd": state_meanstd}

    def _save(self) -> dict[str, Any]:
        """Hook to save a prior's state to disk."""
        return {}

    @classmethod
    def load(cls, state: Mapping[str, Any], map_location=None, **kwargs):
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

        meanstdstate = state["state_meanstd"]
        substate = state["state"]
        for attrname in obj._get_state_attrs():
            try:
                setattr(obj, attrname, MeanStd(**meanstdstate[attrname]))
            except KeyError:
                with suppress(KeyError):
                    setattr(obj, attrname, substate[attrname])
        obj._load(substate, map_location=map_location, **kwargs)
        return obj

    def _load(self, state: Mapping[str, Any], *, map_location=None, **kwargs):
        """Hook to load a prior's state from disk.

        Args:
            state: The saved state.
            n_samples: The number of samples in each group.
            n_features: The number of features in each group.
            map_location: A device to map any potential PyTorch state to.
            **kwargs: Additional, class-specific, arguments.
        """
        pass


def building_docs() -> bool:
    return "MOFAFLEX_DOCS" in os.environ


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


def filter_constant_features(data: "MofaFlexDataset"):
    nonconstantfeatures = {}
    view_vars = data.apply(lambda adata, group_name, view_name: nanvar(adata.X, axis=0), by_group=False)
    threshold = settings.get("eps")
    for view_name, viewvar in view_vars.items():
        nonconst = viewvar > threshold
        _logger.debug(f"Removing {nonconst.size - nonconst.sum()} features from view {view_name}.")
        if issparse(nonconst):
            nonconst = nonconst.toarray()
        nonconstantfeatures[view_name] = data.feature_names[view_name][nonconst]

    data.reindex_features(nonconstantfeatures)


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
                mean = nanmean(arr.tocsr(), axis, keepdims)
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


def default_torch_device(device=None):
    tens = torch.tensor(())
    if device is None:
        return tens.device

    device = torch.device(device)
    try:
        tens.to(device)
    except (RuntimeError, AssertionError):
        default_device = tens.device
        _logger.warning(f"Device {str(device)} is not available. Using default device: {default_device}")
        device = default_device

    return device
