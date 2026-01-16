from collections.abc import Mapping

import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist
from pyro.nn import pyro_method
from torch.nn import functional as F

from ...settings import settings
from .base import LikelihoodWithDispersion


class NegativeBinomial(LikelihoodWithDispersion):
    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        nsamples: Mapping[str, int],
        nfeatures: int,
        sample_means: Mapping[str, NDArray[np.floating]],
        *,
        shift: Mapping[str, NDArray[np.floating]] | None = None,
        init_loc: float = np.e,
        init_scale: float = 0.1,
    ):
        super().__init__(
            view_name, sample_dim, feature_dim, nsamples, nfeatures, init_loc=init_loc, init_scale=init_scale
        )
        self._shift = (
            {group_name: torch.as_tensor(gshift) for group_name, gshift in shift.items()}
            if shift is not None
            else shift
        )
        self._sample_means = {}
        for group_name, gsample_means in sample_means.items():
            shape = gsample_means.shape[0], *((1,) * (abs(self._sample_dim) - 1))
            self._sample_means[group_name] = torch.as_tensor(gsample_means).view(*shape)

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
        if self._shift is not None:
            estimate = estimate + self._shift[group_name][feature_plate.indices[nonmissing_features]]
        rate = F.relu(estimate) * self._sample_means[group_name][sample_plate.indices[nonmissing_samples]]
        return dist.GammaPoisson(
            torch.reciprocal(dispersion), torch.reciprocal(rate * dispersion + settings.get("eps"))
        )
