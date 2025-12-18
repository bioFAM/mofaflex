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
from ..pyro.likelihoods import PyroLikelihood
from ..settings import settings
from ..utils import SaveStateMixin, checked_baseclass

_logger = logging.getLogger(__name__)


class R2(NamedTuple):
    ss_res: float
    ss_tot: float


@checked_baseclass(
    required_init_args=("view_name", "data", "nonnegative"), required_attributes="_priority", registry="dict"
)
class Likelihood(SaveStateMixin, ABC):
    """Base class for MOFA-FLEX likelihoods.

    All likelihood-specific functionality must be implemented via classmethods/staticmethods, subclasses
    must be stateless. Subclasses must also contain two attributes:

        - `_priority`: used during likelihood inference to return the most suitable likelihood
          if multiple likelihoods  are suitable for the given data. Must be non-negative, higher values
          indicate higher priority.
    """

    _state_attrs = ("_view_name", "_nonnegative")

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool = False):
        super().__init__()
        self._view_name = view_name
        self._nonnegative = nonnegative

    def get_pyro_likelihood(self, data: MofaFlexDataset, sample_dim: int, feature_dim: int):
        self._pyro_likelihood = self._get_pyro_likelihood(data, sample_dim, feature_dim)
        return self._pyro_likelihood

    @abstractmethod
    def _get_pyro_likelihood(self, data: MofaFlexDataset, sample_dim: int, feature_dim: int) -> PyroLikelihood:
        """Set up a Pyro likelihood object.

        Args:
            data: The dataset.
            sample_dim: The sample dimension.
            feature_dim: the feature dimension.
        """
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
    def transform_prediction(self, prediction: NDArray[np.floating], group_name: str):
        """Transform the raw model prediction into something compatible with the data, a.k.a. inverse link function.

        Args:
            prediction: The model prediction.
            group_name: The group name.
        """
        pass

    def r2(
        self, y_true: NDArray, y_pred: NDArray[np.floating], group_name: str, alignment_idx: NDArray[int]
    ) -> tuple[float, NDArray[np.floating]]:
        """Calculate R2 (fraction of explained variance) for a factor model.

        Args:
            y_true: The observed data.
            y_pred: The predicted data.
            alignment_idx: Index to use for subsetting arrays aligned to global features in order to align them to local features.
            group_name: The group name.
        """
        r2 = self._r2_impl(y_true, self.transform_prediction(y_pred, group_name), group_name, alignment_idx)
        r2 = max(0.0, 1.0 - r2.ss_res / r2.ss_tot)
        if r2 < settings.get("eps"):
            _logger.warning(
                f"R2 for view {self._view_name} is 0. Increase the number of factors and/or the number of training epochs."
            )
        return r2

    @classmethod
    @property
    def known_likelihoods(cls) -> Mapping[str, type]:
        return MappingProxyType(__class__._registry)
