from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from enum import Enum, auto
from inspect import isabstract
from itertools import chain
from types import MappingProxyType
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd
import pyro
import torch
from numpy.typing import NDArray
from pyro.nn import PyroModule, pyro_method

from ..datasets import MofaFlexDataset
from ..utils import MeanStd, SaveStateMixin, _PyroMeta, checked_baseclass


class APIType(Enum):
    method = auto()
    property = auto()


class API(NamedTuple):
    """Description of a user-facing API attribute."""

    name: str
    """The name of the attribute."""

    type: APIType
    """The type of the attribute (method or property)."""

    has_factors: bool
    """Whether this attribute returns a dict of dataframes with factors."""

    factors_subset: str | None
    """Which property of the object to which this API attribute belongs to query for the subset of factors returned
    by this attribute. Will only be used if `had_factors=True`. If `None`, it is assumed that this attribute returns
    all factors. The property must return a slice or a sequence of indices."""


@checked_baseclass(required_init_args=("names"), registry="dict")
class Prior(SaveStateMixin, ABC, PyroModule, metaclass=_PyroMeta):
    """Base class for MOFA-FLEX factors and weights priors.

    Subclasses can eiher implement `_model` and `_guide`, or reimplment `model` and `guide`. The former set of methods
    operates on one group/view at a time and is convenient for priors without dependencies between groups/views.
    The latter set of methods operates on all groups/views with the respective prior simultaneously, and is useful
    for priors with dependencies between groups/views.

    Subclasses must also implement the `posterior` property to get the summary statistics of the posterior distribution.

    This base class provides default behavior for simple usecases, Subclasses can reimplement any combination of
    methods to customize aspects. Subclasses can also contain two boolean attributs:

        - `_factors`: Indicates whether the subclass is suitable for factors.
        - `_weights`: Indicates whether the subclass is suitable for weights.

    By default, it is assumed that a subclass is suitable for both factors and weights. Generally, specifying these attributes
    should only be necessary if a prior is not suitable for either factors or weights and no wrapper class in _core/priors
    exists.

    Args:
        axis: The axis that the prior is being used for. 0 for factors, 1 for weights.
        names: The names of the groups/views that the prior is responsible for.
        factor_dim: The factor dimension.
        nonfactor_dim: The nonfactor domension. Sample dimension for factors and feature dimension for weights.
    """

    _apilist = []
    _state_attrs = "_names"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not isabstract(cls) and cls.__name__[0] != "_":
            if not cls._factors_allowed and not cls._weights_allowed:
                raise TypeError(f"Class `{cls.__name__}` cannot be used for factors or weights.")

    def __init__(self, names: str | Sequence[str]):
        super().__init__()
        self._names = (names,) if isinstance(names, str) else names

    @classmethod
    def factors_allowed(cls):
        """`True` if this prior can be used for factors."""
        return getattr(cls, "_factors", True)

    @classmethod
    def weights_allowed(cls):
        """`True` if this prior can be used for weights."""
        return getattr(cls, "_weights", True)

    @property
    def names(self) -> tuple[str]:
        """The names of the groups/views that the prior is responsible for."""
        return self._names

    @staticmethod
    def _api(  # noqa: D417
        obj: Callable | property | None = None, *, has_factors: bool | None = None, factors_subset: str | None = None
    ):
        """Mark a method or property as user-facing.

        Subclasses can use this to expose properties or methods to the end user through the main model class.
        If a prior can be used for both factors and weights, the method or property name should contain `a̲x̲i̲s̲`
        (that is the word `axis` with each letter followed by the unicode character U+0332 COMBINING LOW LINE).
        The user-facing method/property will have this replaced by `factor` or `weight` as appropriate.

        Args:
            has_factors: Whether the method/property returns a dict of dataframes with factors. If `True`,
                the user-facing method will have an additional argument `ordered`, which affects whether
                the factors in the dataframes will be ordered by explained variance or not. For this to work,
                the factors must be in the columns. Defaults to `True` for methods and `False` for properties.
                A property with `has_factors=True` will be wrapped in a getter method.
        """

        class __api:
            @staticmethod
            def _add_api(owner, api: API):
                if "_apilist" not in owner.__dict__:
                    owner._apilist = owner._apilist.copy()
                owner._apilist.append(api)

            def __init__(self, func: Callable | property):
                self._func = func

            def __set_name__(self, owner, name: str):
                if isinstance(self._func, Callable):
                    self._add_api(
                        owner,
                        API(name, APIType.method, has_factors if has_factors is not None else True, factors_subset),
                    )
                else:
                    self._add_api(
                        owner,
                        API(name, APIType.property, has_factors if has_factors is not None else False, factors_subset),
                    )
                    self._func.__set_name__(owner, name)
                setattr(owner, name, self._func)

        if obj is not None:
            return __api(obj)
        else:
            return __api

    @classmethod
    def api(cls) -> Iterable[API]:
        """The user-facing API of this prior."""
        return cls._apilist

    @classmethod
    def api_methods(cls) -> Iterable[API]:
        """The user-facing methods of this prior."""
        return (api for api in cls._apilist if api.type == APIType.method)

    @classmethod
    def api_properties(cls) -> Iterable[API]:
        """The user-facing properties of this prior."""
        return (api for api in cls._apilist if api.type == APIType.property)

    def get_datasets(
        self,
        data: MofaFlexDataset,
        axis: Literal[0, 1],
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
    ) -> dict[str, dict[str, pd.DataFrame | np.ndarray]] | None:
        """Hook that is called prior to training.

        If a prior requires any additional covariates during training, it should return a dict of datasets. The keys of
        the dict will be used as argument names for the `model` and `guide` methods of the Pyro prior.

        Args:
            data: The dataset.
            axis: The axis of this prior (0 for samples, 1 for features).
            factor_dim: The factor dimension.
            nonfactor_dim: The nonfactor domension. Sample dimension for factors and feature dimension for weights.
            n_factors: The number of factors.
            n_nonfactors: The number of samples (if `axis == 0`) or features (if `axis == 1`)
        """
        pass

    def adjust_factors(self, data: MofaFlexDataset, axis: Literal[0, 1], factors: list[str]) -> list[str]:
        """Adjust the number and/or names of the factors in the model.

        If a subclass needs to add additional factors to the entire model, this is the place to do it. The subclass should
        store the indices of the factors it added if those need special treatment during training. This is guaranteed to be
        called before `get_datasets`.

        Args:
            data: The dataset.
            axis: The axis of this prior (0 for samples, 1 for features).
            factors: A list of factor names.

        Returns:
            A list of factor names.
        """
        return factors

    def postprocess_results(
        self, results: MeanStd, moment: Literal["mean", "std"] = "mean", name: str | None = None, **kwargs
    ) -> dict[str, NDArray[np.number]] | NDArray[np.number] | None:
        """Hook that is called by the user-facing `get_factors` and `get_weights` methods.

        Subclasses may apply additional postprocessing to the estimated factor and weight values. Any additional arguments in the
        subclass signature will be added to the signature of the user-facing `get_factors`/`get_weights` methods.

        Args:
            results: The factors or weights.
            moment: Which moment the user requested.
            name: Which name (group or view) to postprocess.
            kwargs: Additional arguments.

        Returns:
            If `name is None`, a dict with postprocessed results for all names this prior is responsible for. If `name` is a string
            and this prior is responsible for it,, an array with postprocessed results for that name. Otherwise `None`.
        """
        results = getattr(results, moment)
        if name is not None:
            return results[name] if name in self._names else None
        else:
            return {name: results[name] for name in self._names}

    def on_train_start(
        self,
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] | None = None,
    ):
        """Hook that is called immediately prior to training.

        Subclasses must not reimplement this method. If custom behavior is desired, reimplement `_on_train_start` instead.

        Args:
            factor_dim: The factor dimension.
            nonfactor_dim: The nonfactor domension. Sample dimension for factors and feature dimension for weights.
            n_factors: The number of factors.
            n_nonfactors: The number of samples (if this prior is used for factors) or features (if this prior is used for weights).
            init_tensor: Initialization values.
        """
        self._shapes = {}
        shape = [1] * abs(min(factor_dim, nonfactor_dim))
        shape[factor_dim] = n_factors
        for name in self._names:
            cshape = shape.copy()
            cshape[nonfactor_dim] = n_nonfactors[name]
            self._shapes[name] = tuple(cshape)

        self._squeezedims = tuple(
            i
            for i in chain(
                range(min(factor_dim, nonfactor_dim) + 1, max(factor_dim, nonfactor_dim)),
                range(max(factor_dim, nonfactor_dim) + 1, 0),
            )
        )

        self._on_train_start(factor_dim, nonfactor_dim, n_factors, n_nonfactors, init_tensor)

    def _on_train_start(
        self,
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] | None = None,
    ):
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

    def on_train_end(
        self,
        data: MofaFlexDataset,
        factor_names: Sequence[str],
        nonfactor_names: Mapping[str, Sequence[str]],
        results: MeanStd,
        results_nonnegative: dict[str, bool],
        batch_size: int,
    ):
        """Hook that is called at the end of training.

        Args:
            data: The dataset used during training.
            factor_names: Names of all factors.
            nonfactor_names: Names of the non-factor dimension (sample names or feature names).
            results: The factors or weights.
            results_nonnegative: Whether the factors/weights were constrained to be nonnegative for each group/view.
            batch_size: The batch size used during training.
        """
        pass

    @pyro_method
    def model(
        self, factor_plate: pyro.plate, nonfactor_plates: Mapping[str, pyro.plate], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Pyro model for the prior.

        Args:
            factor_plate: Pyro plate for the factors.
            nonfactor_plates: Pyro plates for the nonfactors (samples or features) for all groups/views.
            **kwargs: Additional arguments that may only be relevant for particular subclasses.

        Returns:
            A dict of sampled tensors for each group/view.
        """
        return {name: self._model(name, factor_plate, nonfactor_plates[name], **kwargs) for name in self._names}

    def _model(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        """Pyro model for the prior.

        Args:
            name: The name of the current group/view.
            factor_plate: Pyro plate for the factors.
            nonfactor_plate: Pyro plate for the nonfactors (samples or features).
            **kwargs: Additional arguments that may only be relevant for particular subclasses.
        """
        raise NotImplementedError

    @pyro_method
    def guide(
        self, factor_plate: Mapping[str, pyro.plate], nonfactor_plates: Mapping[str, pyro.plate], **kwargs
    ) -> dict[str, torch.Tensor]:
        """Pyro guide for the prior.

        Args:
            factor_plate: Pyro plate for the factors.
            nonfactor_plates: Pyro plates for the nonfactors (samples or features) for all groups/views.
            **kwargs: Additional arguments that may only be relevant for particular subclasses.

        Returns:
            A dict of sampled tensors for each group/view.
        """
        return {name: self._guide(name, factor_plate, nonfactor_plates[name], **kwargs) for name in self._names}

    def _guide(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        """Pyro guide for the prior.

        Args:
            name: The name of the current group/view.
            factor_plate: Pyro plate for the factors.
            nonfactor_plate: Pyro plate for the nonfactors (samples or features).
            **kwargs: Additional arguments that may only be relevant for particular subclasses.
        """
        raise NotImplementedError

    @property
    def learning_rate_multipliers(self) -> Iterable[tuple[str, float]]:
        """Multiplicative factors for the base learning rate for individual parameters.

        Returns:
            An iterator yielding two-element tuples with parameter names as the first element and multipliers as the second.
            If a multiplier for a parameter is 1 (i.e. no special learning rate is required), the parameter may be missing
            from the iterator.
        """
        return zip()

    @property
    @abstractmethod
    def posterior(self) -> MeanStd:
        """The estimated factors/weights."""
        pass

    @staticmethod
    def known_priors(filter: Literal["factors", "weights"] | None = None) -> Mapping[str, type["Prior"]]:
        """Get all known priors.

        Args:
            filter: Whether to get only factor or weight priors. Defaults to all priors.
        """
        if filter is not None:
            return {
                name: subcls for name, subcls in __class__._registry.items() if getattr(subcls, f"{filter}_allowed")()
            }
        else:
            return MappingProxyType(__class__._registry)
