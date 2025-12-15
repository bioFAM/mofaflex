from abc import ABC, abstractmethod
from collections.abc import Iterator

from pyro.nn import PyroModule, pyro_method

from ..datasets import CovariatesDataset, MofaFlexDataset
from ..pyro.utils import _PyroMeta


class Term(ABC, PyroModule, metaclass=_PyroMeta):
    @pyro_method
    @abstractmethod
    def model(self, data, sample_idx, nonmissing_samples, nonmissing_features, **kwargs):
        pass

    @pyro_method
    @abstractmethod
    def guide(self, data, sample_idx, nonmissing_samples, nonmissing_features, **kwargs):
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
    def learning_rate_multipliers(self) -> Iterator[tuple[str, float]]:
        """Multiplicative factors for the base learning rate for individual parameters.

        Returns:
            An iterator yielding two-element tuples with parameter names as the first element and multipliers as the second.
            If a multiplier for a parameter is 1 (i.e. no special learning rate is required), the parameter may be missing
            from the iterator.
        """
        return zip()
