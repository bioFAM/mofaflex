from collections.abc import Mapping

import numpy as np
import pyro
import torch
from pyro import distributions as dist
from pyro.nn import pyro_method

from ...settings import settings
from ...utils import Vector
from .base import LikelihoodWithDispersion


class BetaBinomial(LikelihoodWithDispersion):
    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        nsamples: Mapping[str, int],
        nfeatures: int,
        *,
        shift: Mapping[str, Vector[np.floating]] | None = None,
        init_loc: float = 1.0,
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

    @pyro_method
    def _model(
        self,
        id: str,
        data: Mapping[str | None, torch.Tensor],
        estimate: torch.Tensor,
        group_name: str,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ) -> pyro.distributions.Distribution:
        dispersion = self._model_dispersion(
            id, estimate, group_name, sample_plate, feature_plate, nonmissing_samples, nonmissing_features
        )
        if self._shift is not None:
            estimate = estimate + self._shift[group_name][feature_plate.indices[nonmissing_features]]
        estimate = torch.sigmoid(estimate)
        return dist.BetaBinomial(
            dispersion * estimate + settings.eps, dispersion * (1 - estimate) + settings.eps, data["total"]
        ), data["sample"]
