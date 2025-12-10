from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from enum import Enum, auto
from inspect import isabstract, signature
from typing import Any, Literal, NamedTuple

import pandas as pd

from ..datasets import CovariatesDataset, MofaFlexDataset
from ..pyro.priors import Prior as PyroPrior
from ..utils import MeanStd


class _PriorMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        args = list(args)
        if cls == Prior:
            args = args[1:]
        obj.__init__(*args, **kwargs)
        return obj


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


class Prior(metaclass=_PriorMeta):
    """Base class for MOFA-FLEX factors and weights priors.

    Acts as a wrapper around a corresponding Pyro prior to handle additional logic and state, e.g. covariates.
    This base class provides default behavior for simple usecases, Subclasses can reimplement any combination of
    methods to customize aspects. Subclasses must also contain two boolean attributs:

        - `_factors`: Indicates whether the subclass is suitable for factors.
        - `_weights`: Indicates whether the subclass is suitable for weights.

    Args:
        axis: The axis that the prior is being used for. 0/`samples` for factors, 1/`features` for weights.
        names: The names of the groups/views that the prior is responsible for.
    """

    __registry = {}
    _apilist = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not isabstract(cls) and cls.__name__[0] != "_":
            for attr in ("_factors", "_weights"):
                if not hasattr(cls, attr):
                    raise NotImplementedError(f"Class `{cls.__name__}` does not have attribute `{attr}`.")
            if not cls._factors and not cls._weights:
                raise TypeError(f"Class `{cls.__name__}` cannot be used for factors or weights.")
            init_sig = signature(cls.__init__)
            for arg in ("axis", "names"):
                if arg not in init_sig.parameters:
                    raise TypeError(f"Constructor of class `{cls.__name__}` is missing the {arg} argument.")

        if cls._get_pyro_prior is __class__._get_pyro_prior:
            cls.__prior = cls.__name__
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

    def __init__(self, axis: Literal[0, 1, "samples", "features"], names: str | Sequence[str]):
        if isinstance(axis, int):
            self._axis = axis
        else:
            self._axis = 0 if axis == "samples" else 1

        priorname = getattr(self, "__prior", self.__class__.__name__)
        if self._axis == 0 and not getattr(self, "_factors", True):
            raise NotImplementedError(f"The prior {priorname} cannot be used for factors.")
        elif self._axis == 1 and not getattr(self, "_weights", True):
            raise NotImplementedError(f"The prior {priorname} cannot be used for weights.")
        self._names = (names,) if isinstance(names, str) else names

        with suppress(AttributeError):
            for attr in self._state_attrs:
                setattr(self, attr, None)

    @staticmethod
    def class_(name: str) -> _PriorMeta:
        """The the prior class object for a name."""
        try:
            return __class__.__registry[name]
        except KeyError:
            return __class__

    @property
    def axis(self) -> Literal[0, 1]:
        """The axis of this prior."""
        return self._axis

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

    def pyro_prior(self, *args, **kwargs):
        """Get a Pyro prior for this prior.

        This is used by the Pyro model. This method should not be reimplemented by subclasses, if custom behavior
        is required, reimplement `_get_pyro_prior` instead.
        """
        self._pyro_prior = self._get_pyro_prior(*args, **kwargs)
        return self._pyro_prior

    def _get_pyro_prior(self, *args, **kwargs):
        """The default implementation for getting a Pyro prior.

        Defaults to constructing a Pyro prior with the same name as the current prior.
        """
        return PyroPrior(self.__prior, self._names, *args, **kwargs)

    def get_datasets(self, data: MofaFlexDataset) -> dict[str, CovariatesDataset] | None:
        """Hook that is called prior to training.

        If a prior requires any additional covariates during training, it should return a dict of datasets. The keys of
        the dict will be used as argument names for the `model` and `guide` methods of the Pyro prior.

        Args:
            data: The dataset.
        """
        pass

    def adjust_factors(self, factors: list[str]) -> list[str]:
        """Adjust the number and/or names of the factors in the model.

        If a subclass needs to add additional factors to the entire model, this is the place to do it. The subclass should
        store the indices of the factors it added if those need special treatment during training. This is guaranteed to be
        called after `get_datasets`.

        Args:
            factors: A list of factor names.

        Returns:
            A list of factor names.
        """
        return factors

    def postprocess_results(
        self, results: MeanStd, moment: Literal["mean", "std"] = "mean", **kwargs
    ) -> dict[str, pd.DataFrame]:
        """Hook that is called by the user-facing `get_factors` and `get_weights` methods.

        Subclasses may apply additional postprocessing to the estimated factor and weight values. Any additional arguments in the
        subclass signature will be added to the signature of the user-facing `get_factors`/`get_weights` methods.

        Args:
            results: The factors or weights.
            moment: Which moment the user requested.
            kwargs: Additional arguments.
        """
        results = getattr(results, moment)
        return {name: results[name] for name in self._names}

    def on_train_start(self):
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

    def save(self) -> dict[str, Any]:
        """Called by the model to save its state to disk.

        If a subclass has a class attribute `_state_attrs`, which is a sequence of strings, each element of this list is used
        as the name of an instance variable to be saved to disk. Similarly, if a subclass has a class attribute `_state_attrs_meanstd`,
        which is a sequence of strings, each element of this list is assumed to be an instance variable of type `MeanStd` to be saved
        to disk. Subclasses must not reimplement this method. If custom behavior is desired, reimplement `_save` instead.
        """
        state = {}
        if hasattr(self, "_state_attrs"):
            for attr in self._state_attrs:
                state[attr] = getattr(self, attr)
        if hasattr(self, "_state_attrs_meanstd"):
            for attr in self._state_attrs_meanstd:
                state[attr] = getattr(self, attr)._asdict()
        state.update(self._save())
        return {"axis": self._axis, "names": self._names, "class": self.__class__.__name__, "state": state}

    def _save(self) -> dict[str, Any]:
        """Hook to save a prior's state to disk."""
        return {}

    @classmethod
    def load(cls, state: dict[str, Any], n_factors: int, n_nonfactors: Mapping[str, int], map_location=None):
        """Called by the model to restore its state from disk.

        If a subclass has a class attribute `state_attrs`, which is a sequence of strings, each element of this list is used
        as the name of an instance variable to be restored. Similarly, if a subclass has a class attribute `_state_attrs_meanstd`,
        which is a sequence of strings, each element of this list is assumed to be an instance variable of type `MeanStd` to be
        restored.Subclasses must not reimplement this method. If custom behavior is desired, reimplement `_load` instead.

        Args:
            state: The saved state.
            n_factors: The number of factors in the model.
            n_nonfactors: The number of samples (if `self.axis == 0`) or features (if `self.axis == 1`)
            map_location: A device to map any potential PyTorch state to.
        """
        try:
            subcls = __class__.__registry[state["class"]]
            obj = subcls.__new__(subcls)
        except (KeyError, AttributeError):
            obj = __class__.__new__(cls)
        obj._axis = state["axis"]
        obj._names = state["names"]

        substate = state["state"]
        if hasattr(obj, "_state_attrs"):
            for attr in obj._state_attrs:
                setattr(obj, attr, substate.get(attr))
        if hasattr(obj, "_state_attrs_meanstd"):
            for attrname in obj._state_attrs_meanstd:
                if (attr := substate.get(attrname)) is not None:
                    attr = MeanStd(**attr)
                setattr(obj, attrname, attr)
        obj._load(substate, n_factors, n_nonfactors, map_location=map_location)
        return obj

    def _load(self, state, n_factors: int, n_nonfactors: Mapping[str, int], map_location=None):
        """Hook to load a prior's state from disk.

        Args:
            state: The saved state.
            n_factors: The number of factors in the model.
            n_nonfactors: The number of samples (if `self.axis == 0`) or features (if `self.axis == 1`)
            map_location: A device to map any potential PyTorch state to.
        """
        pass

    @staticmethod
    def known_priors(filter: Literal["factors", "weights"] | None = None) -> Sequence[str]:
        """Get all known priors.

        Args:
            filter: Whether to get only factor or weight priors. Defaults to all priors.
        """
        if filter is not None:
            priors = tuple(name for name, subcls in __class__.__registry.items() if getattr(subcls, f"_{filter}"))
        else:
            priors = tuple(__class__.__registry.keys())
        pyropriors = tuple(
            pyroprior for pyroprior in PyroPrior.known_factor_priors() if pyroprior not in __class__.__registry
        )
        return pyropriors + priors
