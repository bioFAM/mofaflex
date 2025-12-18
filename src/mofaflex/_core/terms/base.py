from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from inspect import isabstract

import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro.nn import PyroModule, pyro_method

from ..datasets import CovariatesDataset, MofaFlexDataset
from ..pyro.utils import _PyroMeta
from ..utils import SaveStateMixin


class Term(SaveStateMixin, ABC, PyroModule, metaclass=_PyroMeta):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not isabstract(cls) and cls.__name__[0] != "_":
            __class__._registry[cls.__name__] = cls

    def __new__(cls, *args, **kwargs):
        if cls != __class__ or len(args) == 0 or not isinstance(args[0], str):
            return super().__new__(cls)
        try:
            term = args[0]
            subcls = cls._registry[term]
            return subcls.__new__(subcls, *args[1:], **kwargs)
        except KeyError as e:
            raise NotImplementedError(f"Unknown term {term}.") from e

    @pyro_method
    @abstractmethod
    def model(
        self,
        sample_plates: Mapping[str, pyro.plate],
        feature_plates: Mapping[str, pyro.plate],
        nonmissing_samples: Mapping[str, Mapping[str, torch.Tensor | slice]],
        nonmissing_features: Mapping[str, Mapping[str, torch.Tensor | None]],
        **kwargs,
    ):
        """Pyro model for the term.

        Args:
            sample_plates: Pyro plates for the samples.
            feature_plates: Pyro plates for the features.
            nonmissing_samples: Index tensors indicating which global sample indices of the current minibatch are present
                in the groups and views.
            nonmissing_features: Index tensors indicating which global feature indices of the current minibatch are present
                in the groups and views.
            kwargs: Additional covariates sampled from datasets returned by `get_datasets`.
        """
        pass

    @pyro_method
    @abstractmethod
    def guide(self, nonmissing_samples, nonmissing_features, **kwargs):
        """Pyro guide for the term.

        Args:
            sample_plates: Pyro plates for the samples.
            feature_plates: Pyro plates for the features.
            nonmissing_samples: Index tensors indicating which global sample indices of the current minibatch are present
                in the groups and views.
            nonmissing_features: Index tensors indicating which global feature indices of the current minibatch are present
                in the groups and views.
            kwargs: Additional covariates sampled from datasets returned by `get_datasets`.
        """
        pass

    @abstractmethod
    def predict(
        self, group_name: str, view_name: str, subset_idx: NDArray[int] | slice = slice(None)
    ) -> NDArray[np.floating]:
        pass

    def prediction_components(
        self, group_name: str, view_name: str, subset_idx: NDArray[int] | slice = slice(None)
    ) -> Iterable[tuple[str, NDArray[np.floating]]]:
        pass

    @property
    def component_order(self):
        pass

    @component_order.setter
    def component_order(self, order: NDArray[int]):
        pass

    def get_datasets(self, data: MofaFlexDataset) -> dict[str, CovariatesDataset] | None:
        """Hook that is called prior to training.

        If a prior requires any additional covariates during training, it should return a dict of datasets. The keys of
        the dict will be used as argument names for the `model` and `guide` methods of the Pyro prior.

        Args:
            data: The dataset.
        """
        pass

    def on_train_start(self, data: MofaFlexDataset):
        """Hook that is called immediately prior to training."""
        pass

    def on_train_epoch_start(self, epoch: int):
        """Hook that is called at the beginning of each epoch.

        Args:
            epoch: The current epoch.
        """
        pass

    def on_train_epoch_end(self, epoch: int):
        """Hook that is called at the end of each epoch.

        Args:
            epoch: The current epoch.
        """
        pass

    def on_train_end(self, data: MofaFlexDataset, batch_size: int):
        """Hook that is called at the end of training.

        Args:
            data: The dataset used during training.
            batch_size: The batch size used during training.
        """
        pass

    @property
    def learning_rate_multipliers(self) -> Iterable[tuple[str, float]]:
        """Multiplicative factors for the base learning rate for individual parameters.

        Returns:
            An iterable containing two-element tuples with parameter names as the first element and multipliers as the second.
            If a multiplier for a parameter is 1 (i.e. no special learning rate is required), the parameter may be missing
            from the iterable.
        """
        return zip()

    @property
    @abstractmethod
    def nonnegative(self) -> dict[str, dict[str, bool]]:
        """Whether the term's prediction is constrained to non-negative values for each group and view."""
        pass
