from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from inspect import isabstract, signature
from itertools import chain

import pyro
import torch
from pyro.nn import PyroModule, pyro_method

from ...utils import MeanStd
from ..utils import _PyroMeta

PyroParameterDict = PyroModule[torch.nn.ParameterDict]


class Prior(ABC, PyroModule, metaclass=_PyroMeta):
    """Base class for MOFA-FLEX factors."""

    __registry = {}

    def __init__(
        self, names: Sequence[str], factor_dim: int, nonfactor_dim: int, n_factors: int, n_nonfactors: Mapping[str, int]
    ):
        super().__init__()
        self._names = names

        self._shapes = {}
        shape = [1] * abs(min(factor_dim, nonfactor_dim))
        shape[factor_dim] = n_factors
        for name in names:
            shape[nonfactor_dim] = n_nonfactors[name]
            self._shapes[name] = tuple(shape)

        self._squeezedims = tuple(
            i
            for i in chain(
                range(min(factor_dim, nonfactor_dim) + 1, max(factor_dim, nonfactor_dim)),
                range(max(factor_dim, nonfactor_dim) + 1, 0),
            )
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isabstract(cls) and cls.__name__[0] != "_":
            init_sig = signature(cls.__init__)
            for arg in ("names", "factor_dim", "nonfactor_dim", "n_factors", "n_nonfactors", "kwargs"):
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
            raise NotImplementedError(f"Unknown prior {prior}.") from e

    @pyro_method
    def model(
        self, factor_plate: pyro.plate, nonfactor_plates: Mapping[str, pyro.plate], **kwargs
    ) -> dict[str, torch.Tensor]:
        return {name: self._model(name, factor_plate, nonfactor_plates[name], **kwargs) for name in self._names}

    def _model(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @pyro_method
    def guide(
        self, factor_plate: Mapping[str, pyro.plate], nonfactor_plates: Mapping[str, pyro.plate], **kwargs
    ) -> dict[str, torch.Tensor]:
        return {name: self._guide(name, factor_plate, nonfactor_plates[name], **kwargs) for name in self._names}

    def _guide(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @property
    def learning_rate_multipliers(self) -> dict[str, float]:
        return {}

    @property
    @abstractmethod
    def posterior(self) -> MeanStd:
        pass
