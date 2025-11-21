from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MethodType
from typing import Any, Literal, NamedTuple

import pandas as pd

from ..datasets import CovariatesDataset, MofaFlexDataset
from ..pyro.priors import Prior as PyroPrior


class _PriorMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args[1:], **kwargs)
        return obj


class APIMethod(NamedTuple):
    name: str
    has_factors: bool


class Prior(metaclass=_PriorMeta):
    """Base class for MOFA-FLEX priors."""

    __registry = {}
    _api_methods = []
    _api_properties = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        __class__.__registry[cls.__name__] = cls

    def __new__(cls, *args, **kwargs):
        if cls != __class__ or len(args) == 0 or not isinstance(args[0], str):
            return super().__new__(cls)
        try:
            subcls = cls.__registry[args[0]]
            return subcls.__new__(subcls, *args[1:], **kwargs)
        except KeyError:
            obj = cls.__new__(cls, *args[1:])
            obj.__prior = args[0]
            return obj

    def __init__(self, axis: Literal[0, 1, "samples", "features"], names: str | Sequence[str] | None, **kwargs):
        if isinstance(axis, int):
            self._axis = axis
        else:
            self._axis = 0 if axis == "samples" else 1
        self._names = names if isinstance(names, Sequence) else (names,)

    @property
    def axis(self):
        return self._axis

    def _api(
        obj: Prior | Callable | None = None, attr: MethodType | property | str = None, *, has_factors: bool = True
    ):
        class __api:
            @staticmethod
            def _add_api(owner, api: APIMethod | str):
                if isinstance(api, APIMethod):
                    attr = "_api_metods"
                else:
                    attr = "_api_properties"
                if attr not in owner.__dict__:
                    if isinstance(owner, type):
                        setattr(owner, attr, [])
                    else:
                        setattr(owner, attr, getattr(owner, attr).copy())
                getattr(owner, attr).append(api)

            def __new__(cls, func: Callable | MethodType | property):
                if isinstance(func, MethodType):
                    cls._add_api(func.__self__, APIMethod(func.__name__, has_factors))
                    return None
                else:
                    return super().__new__(cls)

            def __init__(self, func: Callable | property):
                self._func = func

            def __set_name__(self, owner, name: str):
                if isinstance(self._func, Callable):
                    self._add_api(owner, APIMethod(name, has_factors))
                else:
                    self._add_api(owner, name)
                    self._func.__set_name__(owner, name)
                setattr(owner, name, self._func)

        if obj is not None:
            if isinstance(obj, Callable | property):
                return __api(obj)
            elif isinstance(attr, MethodType):
                return __api(attr)
            elif attr is None:
                raise ValueError("need attr if invoked on a Prior instance")

            if "_api_properties" not in obj.__dict__:
                obj._api_properties = obj._api_properties.copy()
            obj._api_properties.append(attr)
            return obj
        else:
            return __api

    @property
    def api_methods(self) -> Sequence[str]:
        return self._api_methods

    @property
    def api_properties(self) -> Sequence[str]:
        return self._api_properties

    def pyro_prior(self, *args, **kwargs):
        return PyroPrior(self.__prior, self._names, *args, **kwargs)

    def get_datasets(self, data: MofaFlexDataset) -> dict[str, CovariatesDataset] | None:
        pass

    def adjust_factors(self, factors: list[str]) -> list[str]:
        return factors

    def on_train_start(self, batch_size: int):
        pass

    def on_train_epoch_start(self, epoch: int):
        pass

    def on_train_epoch_end(self, epoch: int):
        pass

    def on_train_end(
        self,
        data: MofaFlexDataset,
        factor_names: Sequence[str],
        nonfactor_names: Sequence[str],
        results_mean: dict[str, pd.DataFrame],
        results_std: dict[str, pd.DataFrame],
        results_nonnegative: dict[str, bool],
        batch_size: int,
    ):
        pass

    def save(self) -> dict[str, Any]:
        state = {}
        if hasattr(self, "_state_attrs"):
            for attr in self._state_attrs:
                state[attr] = getattr(self, attr)
        state.update(self._save())
        return {"axis": self._axis, "names": self._names, "class": self.__class__.__name__, "state": state}

    def _save(self) -> dict[str, Any]:
        return {}

    @classmethod
    def load(cls, state: dict[str, Any], n_factors: int, n_nonfactors: Mapping[str, int], map_location=None):
        try:
            subcls = __class__.__registry[state["class"]]
            obj = subcls.__new__(subcls)
        except (KeyError, AttributeError):
            obj = __class__.__new__(cls)
        obj._axis = state["axis"]
        obj._names = state["names"]

        substate = state["state"]
        if hasattr(obj, "_state_attrs"):
            for attr in obj._state_attrs:
                setattr(obj, attr, substate.get(attr))
        obj._load(substate, n_factors, n_nonfactors, map_location=map_location)
        return obj

    def _load(self, state, n_factors: int, n_nonfactors: Mapping[str, int], map_location=None):
        pass
