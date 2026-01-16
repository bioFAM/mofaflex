import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from types import MappingProxyType
from typing import NamedTuple

import numpy as np
from anndata import AnnData
from array_api_compat import array_namespace
from numpy.typing import NDArray
from scipy.sparse import issparse

from ..datasets import MofaFlexDataset
from ..utils import SaveStateMixin, checked_baseclass
from .pyro import Likelihood as PyroLikelihood

_logger = logging.getLogger(__name__)


class R2(NamedTuple):
    ss_res: float
    ss_tot: float


@checked_baseclass(
    required_init_args=("view_name", "data", "nonnegative"), required_attributes="_priority", registry="dict"
)
class Likelihood(SaveStateMixin, ABC):
    """Base class for MOFA-FLEX likelihoods.

    Subclasses must contain the `priority` attribute, which is used during likelihood inference to return the
    most suitable likelihood if multiple likelihoods  are suitable for the given data. Must be non-negative,
    higher values indicate higher priority.

    Subclasses must also implement the `_get_pyro_likelihood`, `_validate`, `_r2_impl`, and `transform_prediction`
    methods.

    Args:
        view_name: The name of the view for this likelihood.
        data: The dataset.
        nonnegative: Whether the model prediction for this view is constrained to be nonnegative.
    """

    _state_attrs = ("_view_name", "_nonnegative")

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool = False):
        super().__init__()
        self._view_name = view_name
        self._nonnegative = nonnegative

    def get_pyro_likelihood(self, data: MofaFlexDataset, sample_dim: int, feature_dim: int):
        """Set up a Pyro likelihood object.

        Subclasses must not reimpllement this method, but `_get_pyro_likelihood`.

        Args:
            data: The dataset.
            sample_dim: The sample dimension.
            feature_dim: the feature dimension.
        """
        self._pyro_likelihood = self._get_pyro_likelihood(data, sample_dim, feature_dim)
        return self._pyro_likelihood

    @abstractmethod
    def _get_pyro_likelihood(self, data: MofaFlexDataset, sample_dim: int, feature_dim: int) -> PyroLikelihood:
        pass

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

    def on_train_end(self, data: MofaFlexDataset, batch_size: int):
        """Hook that is called at the end of training.

        Args:
            data: The dataset used during training.
            batch_size: The batch size used during training.
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
    def infer(cls, view: AnnData, *args) -> type["Likelihood"]:
        """Infer a suitable likelihood for the given data.

        Args:
            view: The data.
            *args: Ignored.
        """
        data = view.X.data if issparse(view.X) else view.X
        xp = array_namespace(data)
        data = data[~xp.isnan(data)]

        inferred = {subcls: subcls._priority for subcls in __class__._registry.values() if subcls._validate(data, xp)}
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

    @abstractmethod
    def _r2_impl(
        self, y_true: NDArray, y_pred: NDArray[np.floating], alignment_idx: NDArray[int], group_name: str
    ) -> R2:
        """Implementation of R2 calculation.

        Args:
            y_true: The observed data.
            y_pred: The predicted data.
            alignment_idx: Index to use for subsetting arrays aligned to global features in order to align them to local features.
            group_name: The group name.
        """
        pass

    @abstractmethod
    def transform_prediction(
        self,
        prediction: NDArray[np.floating],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ):
        """Transform the raw model prediction into something compatible with the data, a.k.a. inverse link function.

        Args:
            prediction: The model prediction.
            group_name: The group name.
            sample_idx: The sample indices of the prediction, if only a subset of samples were predicted.
            feature_idx: The feature indices of the prediction, if only a subset of features were predicted.
        """
        pass

    @abstractmethod
    def transform_data(
        self,
        data: NDArray[np.number],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ):
        """Transform the data into something compatible with the raw model prediction, a.k.a. link function.

        Args:
            data: The data.
            group_name: The group name.
            sample_idx: The sample indices of the prediction, if only a subset of samples were predicted.
            feature_idx: The feature indices of the prediction, if only a subset of features were predicted.
        """

    def r2(
        self,
        y_true: NDArray,
        y_pred: NDArray[np.floating],
        group_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
    ) -> tuple[float, NDArray[np.floating]]:
        """Calculate R2 (fraction of explained variance) for a factor model.

        Args:
            y_true: The observed data.
            y_pred: The predicted data.
            group_name: The group name.
            sample_idx: The sample indices of the prediction, if only a subset of samples were predicted.
            feature_idx: The feature indices of the prediction, if only a subset of features were predicted.
        """
        r2 = self._r2_impl(
            y_true,
            self.transform_prediction(y_pred, group_name, sample_idx, feature_idx),
            group_name,
            sample_idx,
            feature_idx,
        )
        return max(0.0, 1.0 - r2.ss_res / r2.ss_tot)

    @classmethod
    def known_likelihoods(cls) -> Mapping[str, type["Likelihood"]]:
        """Get all known likelihoods."""
        return MappingProxyType(__class__._registry)

    @property
    def dispersion(self):
        """Get the estimated dispersion."""
        pass
