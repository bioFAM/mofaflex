from collections.abc import Mapping

import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist

from .base import PyroLikelihood, PyroLikelihoodWithShiftMixin


class PyroBernoulli(PyroLikelihoodWithShiftMixin, PyroLikelihood):
    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        nsamples: Mapping[str, int],
        nfeatures: int,
        shift: Mapping[str, NDArray[np.floating]],
    ):
        super().__init__(view_name, sample_dim, feature_dim, nsamples, nfeatures, shift=shift)

    def _model(
        self,
        estimate: torch.Tensor,
        group_name: str,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ) -> pyro.distributions.Distribution:
        return dist.Bernoulli(logits=estimate + self._get_shift(group_name))

    def _guide(self, group_name: str, sample_plate: pyro.plate, feature_plate: pyro.plate):
        pass
