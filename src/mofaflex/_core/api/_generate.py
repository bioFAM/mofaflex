import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from inspect import Parameter, signature


class APIWrapper(ABC):
    @abstractmethod
    def __call__(self, axis, names):
        pass

    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __eq__(self, other):
        if not isinstance(other, __class__):
            return NotImplemented
        elif other.__class__ != self.__class__:
            return False
        else:
            return self._args == other._args and self._kwargs == other._kwargs

    def __hash__(self):
        return hash((self.__class__, self._args, tuple(sorted(self._kwargs.items()))))


def init_api(module: str, basecls: type, subclss: Mapping[str, type]):
    mod = sys.modules[module]
    coreinit = basecls.__dict__.get("__init__", None)
    if coreinit is not None:
        coresig = signature(coreinit)

    all_ = []

    basewrapper = type(basecls.__name__, (APIWrapper,), {"__module__": module})

    for subname, subcls in subclss.items():
        sig = signature(subcls.__init__)
        annots = subcls.__init__.__annotations__
        if coreinit is not None:
            params = [
                param
                for i, param in enumerate(sig.parameters.values())
                if i == 0 or param.name not in coresig.parameters
            ]
            sig = sig.replace(parameters=params)
            annots = {param.name: param.annotation for param in params if param.annotation is not Parameter.empty}

        def init(self, *args, **kwargs):
            self.__init__.__signature__.bind(self, *args, **kwargs)  # check for argument compatibility
            super(self.__class__, self).__init__(*args, **kwargs)

        def call(self, *args, **kwargs):
            return self._cls(*args, *self._args, **kwargs, **self._kwargs)

        if coreinit is not None:
            call.__signature__ = coresig
            call.__annotations__ = coreinit.__annotations__

        init.__signature__ = sig
        init.__annotations__ = annots
        init.__name__ = "__init__"
        init.__qualname__ = f"{subname}.__init__"
        call.__name__ = "__call__"
        call.__qualname__ = f"{subname}.__call__"
        apicls = type(
            subname, (basewrapper,), {"_cls": subcls, "__init__": init, "__call__": call, "__module__": module}
        )
        apicls.__doc__ = subcls.__doc__

        setattr(mod, subname, apicls)
        all_.append(subname)

    setattr(mod, basewrapper.__name__, basewrapper)
    mod.__all__ = all_
    mod.__dir__ = lambda: all_
