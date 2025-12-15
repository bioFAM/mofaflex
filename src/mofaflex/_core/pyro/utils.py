from abc import ABC

import torch
from pyro.nn import PyroModule

PyroParameterDict = PyroModule[torch.nn.ParameterDict]
PyroModuleDict = PyroModule[torch.nn.ModuleDict]


# https://stackoverflow.com/a/61350480
class _PyroMeta(type(ABC), type(PyroModule)):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        if obj.__class__ is not cls:
            args = args[1:]
        obj.__init__(*args, **kwargs)
        return obj
