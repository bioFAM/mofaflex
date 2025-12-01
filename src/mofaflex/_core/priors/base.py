from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from enum import Enum, auto
from typing import Any, Literal, NamedTuple

from numpy.typing import NDArray

from ..datasets import CovariatesDataset, MofaFlexDataset
from ..pyro.priors import Prior as PyroPrior
from ..utils import MeanStd


class _PriorMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args[1:], **kwargs)
        return obj


class APIType(Enum):
    method = auto()
    property = auto()


class API(NamedTuple):
    name: str
    type: APIType
    has_factors: bool


class Prior(metaclass=_PriorMeta):
    """Base class for MOFA-FLEX priors."""

    __registry = {}
    _apilist = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls._get_pyro_prior is __class__._get_pyro_prior:
            cls.__prior = cls.__name__
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

    def __init__(self, axis: Literal[0, 1, "samples", "features"], names: str | Sequence[str], **kwargs):
        if isinstance(axis, int):
            self._axis = axis
        else:
            self._axis = 0 if axis == "samples" else 1
        self._names = names if isinstance(names, Sequence) else (names,)

        with suppress(AttributeError):
            for attr in self._state_attrs:
                setattr(self, attr, None)

    @property
    def axis(self):
        return self._axis

    @staticmethod
    def _api(obj: Callable | property | None = None, *, has_factors: bool | None = None):
        class __api:
            @staticmethod
            def _add_api(owner, api: API):
                if "_apilist" not in owner.__dict__:
                    owner._apilist = owner._apilist.copy()
                owner._apilist.append(api)

            def __init__(self, func: Callable | property):
                self._func = func

            def __set_name__(self, owner, name: str):
                if isinstance(self._func, Callable):
                    self._add_api(owner, API(name, APIType.method, has_factors if has_factors is not None else True))
                else:
                    self._add_api(owner, API(name, APIType.property, has_factors if has_factors is not None else False))
                    self._func.__set_name__(owner, name)
                setattr(owner, name, self._func)

        if obj is not None:
            return __api(obj)
        else:
            return __api

    @property
    def api(self) -> Iterable[API]:
        return self._apilist

    @property
    def api_methods(self) -> Iterable[API]:
        return (api for api in self._apilist if api.type == APIType.method)

    @property
    def api_properties(self) -> Iterable[API]:
        return (api for api in self._apilist if api.type == APIType.property)

    def pyro_prior(self, *args, **kwargs):
        self._pyro_prior = self._get_pyro_prior(*args, **kwargs)
        return self._pyro_prior

    def _get_pyro_prior(self, *args, **kwargs):
        return PyroPrior(self.__prior, self._names, *args, **kwargs)

    def get_datasets(self, data: MofaFlexDataset) -> dict[str, CovariatesDataset] | None:
        pass

    def adjust_factors(self, factors: list[str]) -> list[str]:
        return factors

    def postprocess_results(
        self, results: MeanStd, moment: Literal["mean", "std"] = "mean", **kwargs
    ) -> dict[str, NDArray]:
        results = getattr(results, moment)
        return {name: results[name] for name in self._names}

    def on_train_start(self, batch_size: int):
        pass

    def on_train_epoch_start(self, epoch: int):
        pass

    def on_train_epoch_end(self, epoch: int):
        pass

    def on_train_end(
        self, data: MofaFlexDataset, results: MeanStd, results_nonnegative: dict[str, bool], batch_size: int
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

    @staticmethod
    def known_factor_priors() -> tuple[str]:
        pyropriors = PyroPrior.known_factor_priors()
        priors = tuple(
            name
            for name, subcls in __class__.__registry.items()
            if name not in pyropriors and getattr(subcls, "_factors", False)
        )
        return pyropriors + priors

    @staticmethod
    def known_weight_priors() -> tuple[str]:
        pyropriors = PyroPrior.known_weight_priors()
        priors = tuple(
            name
            for name, subcls in __class__.__registry.items()
            if name not in pyropriors and getattr(subcls, "_weights", False)
        )
        return pyropriors + priors
