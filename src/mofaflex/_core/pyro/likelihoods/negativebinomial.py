import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist
from pyro.nn import pyro_method
from torch.nn import functional as F

from ...settings import settings
from .base import PyroLikelihoodWithDispersion


class PyroNegativeBinomial(PyroLikelihoodWithDispersion):
    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        sample_means: dict[str, dict[str, NDArray[np.floating]]],
        feature_means: dict[str, dict[str, NDArray[np.floating]]],
        *,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
    ):
        super().__init__(
            view_name, sample_dim, feature_dim, sample_means, feature_means, init_loc=init_loc, init_scale=init_scale
        )

        for group_name, gsample_means in sample_means.items():
            shape = self._nsamples[group_name], *((1,) * (abs(self._sample_dim) - 1))
            self.register_buffer(f"_sample_means_{group_name}", torch.as_tensor(gsample_means[view_name]).view(*shape))

    def _get_sample_means(self, group_name: str):
        return getattr(self, f"_sample_means_{group_name}", None)

    @pyro_method
    def _model(
        self,
        estimate: torch.Tensor,
        group_name: str,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ) -> pyro.distributions.Distribution:
        dispersion = self._model_dispersion(
            estimate, group_name, sample_plate, feature_plate, nonmissing_samples, nonmissing_features
        )
        rate = F.relu(estimate) * self._get_sample_means(group_name)[sample_plate.indices[nonmissing_samples]]
        return dist.GammaPoisson(
            torch.reciprocal(dispersion), torch.reciprocal(rate * dispersion + settings.get("eps"))
        )
