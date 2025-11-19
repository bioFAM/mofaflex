from collections.abc import Mapping, Sequence
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
    _apilist = ()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        __class__.__registry[cls.__name__] = cls
        if "_apilist" in cls.__dict__:
            cls._apilist = tuple(cls._apilist)

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

    def _api(func=None, *, has_factors=True):
        class __api:
            def __init__(self, func):
                self._func = func

            def __set_name__(self, owner, name):
                if "_apilist" not in owner.__dict__:
                    owner._apilist = []
                owner._apilist.append(APIMethod(name, has_factors))
                setattr(owner, name, self._func)

        if func is not None:
            return __api(func)
        else:
            return __api

    @classmethod
    def api(cls):
        return cls._apilist

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
