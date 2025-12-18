from abc import ABC

import torch
from pyro.nn import PyroModule

PyroParameterDict = PyroModule[torch.nn.ParameterDict]
PyroModuleDict = PyroModule[torch.nn.ModuleDict]


# https://stackoverflow.com/a/61350480
class _PyroMeta(type(ABC), type(PyroModule)):
    pass
