from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from inspect import isabstract, signature
from itertools import chain

import pyro
import torch
from pyro.nn import PyroModule, pyro_method

from ...utils import MeanStd
from ..utils import _PyroMeta


class Prior(ABC, PyroModule, metaclass=_PyroMeta):
    """Base class for MOFA-FLEX factors and weights priors used in the Pyro model.

    Subclasses can eiher implement `_model` and `_guide`, or reimplment `model` and `guide`. The former set of methods
    operates on one group/view at a time and is convenient for priors without dependencies between groups/views.
    The latter set of methods operates on all groups/views with the respective prior simultaneously, and is useful
    for priors with dependencies between groups/views.

    Subclasses must also implement the `posterior` property to get the summary statistics of the posterior distribution.
    The constructor of subclasses must take a `**kwargs` argument which is ignored. This ensures that users can simply call
    `Prior(distribution, args)`, where args may be a union of arguments suitable for different priors, only a subset
    of which will be used by the concrete Prior. Subclasses must also contain two boolean attributes:

        - `_factors`: Indicates whether the subclass is suitable for factors.
        - `_weights`: Indicates whether the subclass is suitable for weights.

    Args:
        names: Names of groups/views that have the respective distribution.
        factor_dim: The factor dimension.
        nonfactor_dim: The nonfactor domension. Sample dimension for factors and feature dimension for weights.
        n_factors: Number of factors.
        n_nonfactors: Number of nonfactors (samples / features).
    """

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
            for attr in ("_factors", "_weights"):
                if not hasattr(cls, attr):
                    raise NotImplementedError(f"Class `{cls.__name__}` does not have attribute `{attr}`.")
            if not cls._factors and not cls._weights:
                raise TypeError(f"Class `{cls.__name__}` cannot be used for factors or weights.")
            init_sig = signature(cls.__init__)
            for arg in ("names", "factor_dim", "nonfactor_dim", "n_factors", "n_nonfactors", "kwargs"):
                if arg not in init_sig.parameters:
                    raise TypeError(f"Constructor of class `{cls.__name__}` is missing the {arg} argument.")

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
    def learning_rate_multipliers(self) -> dict[str, float]:
        """Multiplicative factors for the base learning rate for individual parameters.

        Returns:
            A dictionary with parameter names as keys and multipliers as values. If a multiplier for a parameter is 1
            (i.e. no special learning rate is required), the parameter may be missing from the returned dictionary.
        """
        return {}

    @property
    @abstractmethod
    def posterior(self) -> MeanStd:
        """The estimated factors/weights."""
        pass

    @staticmethod
    def known_factor_priors() -> tuple[str]:
        """Get all known factor priors."""
        return tuple(name for name, subcls in __class__.__registry.items() if subcls._factors)

    @staticmethod
    def known_weight_priors() -> tuple[str]:
        """Get all known weight priors."""
        return tuple(name for name, subcls in __class__.__registry.items() if subcls._weights)
