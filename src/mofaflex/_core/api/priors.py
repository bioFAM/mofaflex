from abc import ABC, abstractmethod
from collections.abc import Sequence
from inspect import Parameter, signature
from types import MappingProxyType
from typing import Literal

from ..priors import Prior as PriorCore


class Prior(ABC):
    @abstractmethod
    def __call__(self, axis, names):
        pass

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = MappingProxyType(kwargs)

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        elif other.__class__ != self.__class__:
            return False
        else:
            return self._args == other._args and self._kwargs == other._kwargs

    def __hash__(self):
        return hash((self.__class__, self._args, tuple(sorted(self._kwargs.items()))))


__all__ = []


def _init_priors():
    for priorname in PriorCore.known_priors():
        priorcls = PriorCore.class_(priorname)
        sig = signature(priorcls.__init__)
        params = [param for param in sig.parameters.values() if param.name not in ("axis", "names")]
        sig = sig.replace(parameters=params)

        def init(self, *args, **kwargs):
            self.__init__.__signature__.bind(self, *args, **kwargs)  # check for argument compatibility
            super(self.__class__, self).__init__(*args, **kwargs)

        if priorcls is not PriorCore:

            def call(self, axis: Literal[0, 1, "samples", "features"], names: str | Sequence[str]):
                return self._cls(axis, names, *self._args, **self._kwargs)
        else:

            def call(self, axis: Literal[0, 1, "samples", "features"], names: str | Sequence[str]):
                return PriorCore(self.__class__.__name__, axis, names, *self._args, **self._kwargs)

        init.__signature__ = sig
        init.__annotations__ = {
            param.name: param.annotation for param in params if param.annotation is not Parameter.empty
        }
        init.__name__ = "__init__"
        init.__qualname__ = f"{priorname}.__init__"
        call.__name__ = "__call__"
        call.__qualname__ = f"{priorname}.__call__"
        apicls = type(
            priorname, (Prior,), {"_cls": priorcls, "__init__": init, "__call__": call, "__module__": __name__}
        )
        if priorcls is not PriorCore:
            apicls.__doc__ = priorcls.__doc__

        globals()[priorname] = apicls
        __all__.append(priorname)


_init_priors()


def __dir__():
    return __all__
