from collections.abc import Sequence
from typing import Literal

from ..datasets import CovariatesDataset, MofaFlexDataset
from ..pyro.priors import Prior as PyroPrior


class _PriorMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        obj.__init__(*args[1:], **kwargs)
        return obj


class Prior(metaclass=_PriorMeta):
    """Base class for MOFA-FLEX priors."""

    __registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
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

    def __init__(self, axis: Literal[0, 1, "samples", "features"], names: str | Sequence[str] | None, **kwargs):
        if isinstance(axis, int):
            self._axis = axis
        else:
            self._axis = 0 if axis == "samples" else 1
        self._names = names if isinstance(names, Sequence) else (names,)

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

    def on_train_end(self, data: MofaFlexDataset, batch_size: int):
        pass
