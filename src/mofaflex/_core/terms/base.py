from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType, MethodType

import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro.nn import PyroModule, pyro_method

from ..datasets import CovariatesDataset, MofaFlexDataset
from ..utils import PyroMeta, SaveStateMixin, checked_baseclass


class _class_and_instancemethod:
    def __init__(self, func):
        self._func = func
        self._clsfunc = classmethod(func)

    def __get__(self, instance, owner):
        obj = self._func if instance is not None else self._clsfunc
        return obj.__get__(instance, owner)


@checked_baseclass(registry="dict")
class Term(SaveStateMixin, ABC, PyroModule, metaclass=PyroMeta):
    r"""Base class for MOFA-FLEX additive terms.

    A Term represents one additive contribution to the generative model, i.e. a component of the form:

    .. math::
        Y = \Sum_t \text{Term}_t

    Subclasses must implement a Pyro model and guide in the `model` and `guide` methods, respectively, as well as
    the `nonnegative` method. Method or properties that should be exposed to the end user must be marked with `@Term._api`.
    """

    _apilist = []

    @_class_and_instancemethod
    def api(self) -> Iterable[str]:
        """The user-facing API of this term."""
        return self._apilist

    @_class_and_instancemethod
    def api_methods(self) -> Iterable[str]:
        """The user-facing methods of this term."""
        return (api for api in self._apilist if not isinstance(getattr(self.__class__, api), property))

    @_class_and_instancemethod
    def api_properties(self) -> Iterable[str]:
        """The user-facing properties of this prior."""
        return (api for api in self._apilist if isinstance(getattr(self.__class__, api), property))

    def _api(obj: Callable | property | Term | type[Term], attr: MethodType | property | str | None = None):
        """Mark a method or property as user-facing.

        Subclasses can use this to expose properties or methods to the end user through the main model class.
        """

        def _add_api(owner, api: str):
            if "_apilist" not in owner.__dict__:
                owner._apilist = owner._apilist.copy()
            owner._apilist.append(api)

        class __api:
            def __new__(cls, func: Callable | MethodType | property):
                if isinstance(func, MethodType):
                    _add_api(func.__self__, func.__name__)
                    return None
                else:
                    return super().__new__(cls)

            def __init__(self, func: Callable | property):
                self._func = func
                if isinstance(func, property):
                    self.setter = self._setter
                    self.deleter = self._deleter

            def __set_name__(self, owner, name: str):
                _add_api(owner, name)
                setattr(owner, name, self._func)

            def _setter(self, func):
                self._func = self._func.setter(func)
                return self

            def _deleter(self, func):
                self._func = self._func.deleter(func)
                return self

        if isinstance(obj, Callable | property) and not isinstance(obj, __class__) and not isinstance(obj, type):
            return __api(obj)
        elif isinstance(attr, MethodType):
            return __api(attr)
        elif attr is None:
            raise ValueError("Need attr if invoked on a Term instance.")
        _add_api(obj, attr)
        return obj

    @pyro_method
    @abstractmethod
    def model(
        self,
        id: str,
        sample_plates: Mapping[str, pyro.plate],
        feature_plates: Mapping[str, pyro.plate],
        nonmissing_samples: Mapping[str, Mapping[str, torch.Tensor | slice]],
        nonmissing_features: Mapping[str, Mapping[str, torch.Tensor | slice]],
        **kwargs,
    ):
        """Pyro model for the term.

        This method should define all latent variables associated with the term. It should contribute additively to
        the model's overall observation model. Importantly, it must not assume it is the only term.

        Args:
            id: ID to be used in Pyro sample site names to make them unique if multiple additive terms are used.
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
    def guide(
        self,
        id: str,
        nonmissing_samples: Mapping[str, Mapping[str, torch.Tensor | slice]],
        nonmissing_features: Mapping[str, Mapping[str, torch.Tensor | slice]],
        **kwargs,
    ):
        """Pyro guide for the term.

        This method defines the variational distribution for all latent variables associated with the term.

        Args:
            id: ID to be used in Pyro sample site names to make them unique if multiple additive terms are used.
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
        self,
        group_name: str,
        view_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
        idx_cartesian_product: bool = True,
    ) -> NDArray[np.floating]:
        """Predict the value of the term for a given group and view.

        Args:
            group_name: The group.
            view_name: The view.
            sample_idx: The subset of samples to predict for.
            feature_idx: The subset of features to predict for.
            idx_cartesian_product: If both `sample_idx` and `feature_idx` are given, whether to generate predictions
                at the cartesian product of `sample_idx` and `feature_idx`, or at individual positions
                `[sample_idx[i], feature_idx[i]]`. If `False`, `sample_idx` and `feature_idx` must have the same length.
        """
        pass

    def prediction_components(
        self,
        group_name: str,
        view_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
        idx_cartesian_product: bool = True,
    ) -> Iterable[tuple[str, NDArray[np.floating]]]:
        """Predict individual components of this term.

        If the term itself has some additive components, e.g. factors in a factor model, predict each component individually.

        Args:
            group_name: The group.
            view_name: The view.
            sample_idx: The subset of samples to predict for.
            feature_idx: The subset of features to predict for.
            idx_cartesian_product: If both `sample_idx` and `feature_idx` are given, whether to generate predictions
                at the cartesian product of `sample_idx` and `feature_idx`, or at individual positions
                `[sample_idx[i], feature_idx[i]]`. If `False`, `sample_idx` and `feature_idx` must have the same length.
        """
        pass

    @property
    def component_order(self) -> NDArray[int]:
        """Ordering of individual components of this term.

        If the term itself has some additive components, e.g. factors in a factor model, this property specifies the ordering
        of those components, for example by explained variance.
        """
        pass

    @component_order.setter
    def component_order(self, order: NDArray[int]):
        pass

    def get_datasets(
        self, data: MofaFlexDataset, sample_plate_dim: int, feature_plate_dim: int
    ) -> dict[str, CovariatesDataset] | None:
        """Hook that is called prior to training.

        If a prior requires any additional covariates during training, it should return a dict of datasets. The keys of
        the dict will be used as argument names for the `model` and `guide` methods of the Pyro prior.

        Args:
            data: The dataset.
            sample_plate_dim: The sample dimension.
            feature_plate_dim: The feature dimension.
        """
        pass

    def on_train_start(self, data: MofaFlexDataset, sample_plate_dim: int, feature_plate_dim: int):
        """Hook that is called immediately prior to training.

        Args:
            data: The dataset.
            sample_plate_dim: The sample dimension.
            feature_plate_dim: The feature dimension.
        """
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

    @staticmethod
    def known_terms() -> Mapping[str, type[Term]]:
        """Get all known terms."""
        return MappingProxyType(__class__._registry)
