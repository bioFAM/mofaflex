from collections.abc import Mapping

import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist

from .base import Likelihood


class Bernoulli(Likelihood):
    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        nsamples: Mapping[str, int],
        nfeatures: int,
        *,
        shift: Mapping[str, NDArray[np.floating]] | None = None,
    ):
        super().__init__(view_name, sample_dim, feature_dim, nsamples, nfeatures)
        self._shift = (
            {group_name: torch.as_tensor(gshift) for group_name, gshift in shift.items()}
            if shift is not None
            else shift
        )

    def _model(
        self,
        estimate: torch.Tensor,
        group_name: str,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ) -> pyro.distributions.Distribution:
        if self._shift is not None:
            estimate = estimate + self._shift[group_name][feature_plate.indices[nonmissing_features]]
        return dist.Bernoulli(logits=estimate)

    def _guide(self, group_name: str, sample_plate: pyro.plate, feature_plate: pyro.plate):
        pass
