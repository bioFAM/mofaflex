import inspect
import logging
from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np
from anndata import AnnData
from array_api_compat import array_namespace
from numpy.typing import NDArray
from scipy.sparse import issparse

from ..pyro.likelihoods import PyroLikelihood
from ..settings import settings

_logger = logging.getLogger(__name__)


class R2(NamedTuple):
    ss_res: float
    ss_tot: float


class _LikelihoodMeta(type(ABC)):
    def __eq__(cls, o):
        if isinstance(o, str):
            return cls.__name__ == o
        else:
            return super().__eq__(o)

    def __hash__(self):
        return super().__hash__()

    def __str__(cls):
        return cls.__name__


class Likelihood(ABC, metaclass=_LikelihoodMeta):
    """Base class for MOFA-FLEX likelihoods.

    All likelihood-specific functionality must be implemented via classmethods/staticmethods, subclasses
    must be stateless. Subclasses must also contain two attributes:

        - `_priority`: used during likelihood inference to return the most suitable likelihood
          if multiple likelihoods  are suitable for the given data. Must be non-negative, higher values
          indicate higher priority.
        - `scale_data`: indicates whether the likelihood requires data to be centered and scaled to unit variance.
    """

    __subclasses = {}

    def __init_subclass__(cls, **kwargs):
        for attr in ("_priority", "scale_data"):
            if not hasattr(cls, attr):
                raise NotImplementedError(f"Class `{cls.__name__}` does not have attribute `{attr}`.")
        for name, val in inspect.getmembers_static(cls):
            if inspect.isfunction(val):
                raise TypeError(
                    f"Class `{cls.__name__}` must be stateless, but method `{name}` is not static and not a class method."
                )
        super().__init_subclass__(**kwargs)
        __class__.__subclasses[str(cls)] = cls

    @staticmethod
    def known_likelihoods() -> tuple[str]:
        """Get all known likelihoods."""
        return tuple(__class__.__subclasses.keys())

    @staticmethod
    def get(name: str) -> _LikelihoodMeta:
        """Get the likelihood based on its name.

        Args:
            name: The name of the likelihood.
        """
        try:
            return __class__.__subclasses[name]
        except KeyError as e:
            raise NotImplementedError(f"Unknown likelihood `{name}`") from e

    @classmethod
    @abstractmethod
    def pyro_likelihood(
        cls,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        sample_means: dict[str, dict[str, NDArray[np.floating]]],
        feature_means: dict[str, dict[str, NDArray[np.floating]]],
        **kwargs,
    ) -> PyroLikelihood:
        """Set up a Pyro likelihood object.

        Args:
            view_name: The view name.
            sample_dim: The sample dimension.
            feature_dim: the feature dimension.
            sample_means: Averages of samples across features for each group and view.
            feature_means: Averages of features across samples for each group and view.
            **kwargs: Additional arguments, e.g. initialization of the variational parameters.
        """
        pass

    @classmethod
    @abstractmethod
    def _validate(cls, data: NDArray, xp) -> bool:
        """Validate that the current likelihood is suitable for the given data.

        Args:
            data: The data.
            xp: The array-API namespace for the given data.
        """
        pass

    @classmethod
    def _format_validate_exception(cls, view_name: str) -> str:
        return view_name

    @classmethod
    def validate(cls, view: AnnData, group_name: str, view_name: str):
        """Validate that the current likelihood is suitable for the given data.

        Args:
            view: The data.
            group_name: The group name.
            view_name: The view name.
        """
        data = view.X.data if issparse(view.X) else view.X
        xp = array_namespace(data)
        data = data[~xp.isnan(data)]

        if not cls._validate(data, xp):
            raise ValueError(cls._format_validate_exception(view_name))

    @classmethod
    def infer(cls, view: AnnData, *args) -> _LikelihoodMeta:
        """Infer a suitable likelihood for the given data.

        Args:
            view: The data.
            *args: Ignored.
        """
        data = view.X.data if issparse(view.X) else view.X
        xp = array_namespace(data)
        data = data[~xp.isnan(data)]

        inferred = {subcls: subcls._priority for subcls in cls.__subclasses.values() if subcls._validate(data, xp)}
        lklhdcls = max(((subcls, prio) for subcls, prio in inferred.items()), key=lambda x: x[1])[0]
        return lklhdcls

    @staticmethod
    def _Vprime(mu: NDArray[np.floating], nu2: float, nu1: float):
        return 2 * nu2 * mu + nu1

    @classmethod
    def _dV_square(cls, a: NDArray[np.floating], b: NDArray[np.floating], nu2: float, nu1: float):
        # this is based on Zhang: A Coefficient of Determination for Generalized Linear Models (2017)
        dVb = cls._Vprime(b, nu2, nu1)
        dVa = cls._Vprime(a, nu2, nu1)
        sVb = np.sqrt(1 + dVb**2)
        sVa = np.sqrt(1 + dVa**2)
        return 1 / (16 * nu2**2) * (np.log((dVb + sVb) / (dVa + sVa)) + dVb * sVb - dVa * sVa) ** 2

    @classmethod
    @abstractmethod
    def _r2_impl(
        cls,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> R2:
        """Implementation of R2 calculation.

        Args:
            y_true: The observed data.
            y_pred: The predicted data.
            dispersions: The estimated dispersions.
            sample_means: Averages of samples across features.
        """
        pass

    @classmethod
    @abstractmethod
    def transform_prediction(cls, prediction: NDArray[np.floating], sample_means: NDArray[np.floating]):
        """Transform the raw model prediction into something compatible with the data, a.k.a. inverse link function.

        Args:
            prediction: The model prediction.
            sample_means: Averages of samples across features
        """
        pass

    @classmethod
    def _r2_impl_wrapper(
        cls,
        y_true: NDArray,
        factor: NDArray[np.floating],
        weights: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ):
        y_pred = cls.transform_prediction(factor @ weights, sample_means)
        r2 = cls._r2_impl(y_true, y_pred, dispersions, sample_means)
        return max(0.0, 1.0 - r2.ss_res / r2.ss_tot)

    @classmethod
    def _r2(
        cls,
        r2_full: float,
        y_true: NDArray,
        factors: NDArray[np.floating],
        weights: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> NDArray[np.float32]:
        r2s = np.empty(factors.shape[1], dtype=np.float32)
        # For models with a link function that is not the identity, such as Bernoulli, calculating R2 of single
        # factors leads to erroneous results, in the case of Bernoulli it can lead to every factor having negative
        # R2 values. This is because an unimportant factor will not contribute much to the full model, but the zero
        # prediction of this single factor will be mapped by the link function to a non-zero value, which can result
        # in a worse prediction than the intercept-only null model. As a workaround, we therefore calculate R2 of
        # a model with all factors except for one and subtract it from the R2 value of the full model to arrive at
        # the R2 of the current factor.
        for k in range(factors.shape[1]):
            cfactors = np.delete(factors, k, 1)
            cweights = np.delete(weights, k, 0)
            cr2 = cls._r2_impl_wrapper(y_true, cfactors, cweights, dispersions, sample_means)
            r2s[k] = max(0.0, r2_full - cr2)
        return r2s

    @classmethod
    def r2(
        cls,
        view_name: str,
        y_true: NDArray,
        factors: NDArray[np.floating],
        weights: NDArray[np.floating],
        dispersions: NDArray[np.floating],
        sample_means: NDArray[np.floating],
    ) -> tuple[float, NDArray[np.float32]]:
        """Calculate R2 (fraction of explained variance) for a factor model.

        Args:
            view_name: The name of the view.
            y_true: The observed data.
            factors: The factors.
            weights: The weights.
            dispersions: Estimated dispersions.
            sample_means: Averages of samples across features.
        """
        r2_full = cls._r2_impl_wrapper(y_true, factors, weights, dispersions, sample_means)
        if r2_full < settings.get("eps"):
            _logger.warning(
                f"R2 for view {view_name} is 0. Increase the number of factors and/or the number of training epochs."
            )
            return r2_full, np.zeros(factors.shape[1], dtype=np.float32)
        return r2_full, cls._r2(r2_full, y_true, factors, weights, dispersions, sample_means)
