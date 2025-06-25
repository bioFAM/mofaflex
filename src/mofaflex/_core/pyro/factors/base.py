from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from inspect import isabstract, signature

import pyro
import torch
from pyro.nn import PyroModule, pyro_method

from ...utils import MeanStd
from ..utils import _PyroMeta

PyroParameterDict = PyroModule[torch.nn.ParameterDict]


class Factor(ABC, PyroModule, metaclass=_PyroMeta):
    """Base class for MOFA-FLEX factors."""

    __registry = {}

    def __init__(
        self, group_names: Sequence[str], factor_dim: int, sample_dim: int, n_factors: int, n_samples: Mapping[str, int]
    ):
        super().__init__()
        self._group_names = group_names

        self._shapes = {}
        shape = [1] * abs(min(factor_dim, sample_dim))
        shape[factor_dim] = n_factors
        for group_name in group_names:
            shape[sample_dim] = n_samples[group_name]
            self._shapes[group_name] = tuple(shape)

        self._squeezedims = tuple(i for i in range(min(factor_dim, sample_dim) + 1, max(factor_dim, sample_dim)))

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isabstract(cls) and cls.__name__[0] != "_":
            init_sig = signature(cls.__init__)
            for arg in ("group_names", "factor_dim", "sample_dim", "n_factors", "n_samples", "kwargs"):
                if arg not in init_sig.parameters:
                    raise TypeError(f"Constructor of class {cls} is missing the {arg} argument.")

            __class__.__registry[cls.__name__] = cls

    def __new__(cls, prior: str, *args, **kwargs):
        if cls != __class__:
            return super().__new__(cls)
        try:
            subcls = cls.__registry[prior]
            return subcls.__new__(subcls, None, *args, **kwargs)
        except KeyError as e:
            raise NotImplementedError(f"Unknown factor prior {prior}.") from e

    @pyro_method
    def model(
        self, factor_plate: pyro.plate, sample_plates: Mapping[str, pyro.plate], **kwargs
    ) -> dict[str, torch.Tensor]:
        return {
            group_name: self._model(group_name, factor_plate, sample_plates[group_name], **kwargs)
            for group_name in self._group_names
        }

    def _model(self, group_name: str, factor_plate: pyro.plate, sample_plate: pyro.plate, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @pyro_method
    def guide(
        self, factor_plate: Mapping[str, pyro.plate], sample_plates: Mapping[str, pyro.plate], **kwargs
    ) -> dict[str, torch.Tensor]:
        return {
            group_name: self._guide(group_name, factor_plate, sample_plates[group_name], **kwargs)
            for group_name in self._group_names
        }

    def _guide(self, group_name: str, factor_plate: pyro.plate, sample_plate: pyro.plate, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def posterior(self) -> MeanStd:
        pass
