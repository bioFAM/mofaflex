# https://stackoverflow.com/a/61350480
from abc import ABC

import torch
from pyro.nn import PyroModule

PyroParameterDict = PyroModule[torch.nn.ParameterDict]
PyroModuleDict = PyroModule[torch.nn.ModuleDict]


class _PyroMeta(type(ABC), type(PyroModule)):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        args = list(args)
        if obj.__class__ is not cls:
            args = args[1:]
        obj.__init__(*args, **kwargs)
        return obj
