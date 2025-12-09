from abc import ABC, abstractmethod
from collections.abc import Sequence
from inspect import Parameter, signature
from typing import Literal

from ..priors import Prior as PriorCore


class Prior(ABC):
    @abstractmethod
    def __call__(self, axis, names):
        pass


__all__ = []


def _init_priors():
    for priorname in PriorCore.known_priors():
        priorcls = PriorCore.class_(priorname)
        sig = signature(priorcls.__init__)
        params = [param for param in sig.parameters.values() if param.name not in ("axis", "names")]
        sig = sig.replace(parameters=params)

        def init(self, *args, **kwargs):
            self.__init__.__signature__.bind(self, *args, **kwargs)  # check for argument compatibility

            self._args = args
            self._kwargs = kwargs

        def call(self, axis: Literal[0, 1, "samples", "features"], names: str | Sequence[str]):
            return self._cls(axis, names, *self._args, **self._kwargs)

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
