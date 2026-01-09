from collections.abc import Mapping

import pyro
import torch
from pyro import distributions as dist

from .base import Likelihood


class Categorical(Likelihood):
    def __init__(self, view_name: str, sample_dim: int, feature_dim: int, nsamples: Mapping[str, int], nfeatures: int):
        super().__init__(view_name, sample_dim, feature_dim, nsamples, nfeatures)

    def _model(
        self,
        estimate: torch.Tensor,
        group_name: str,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ) -> pyro.distributions.Distribution:
        return dist.Categorical(logits=estimate)

    def _guide(self, group_name: str, sample_plate: pyro.plate, feature_plate: pyro.plate):
        pass
