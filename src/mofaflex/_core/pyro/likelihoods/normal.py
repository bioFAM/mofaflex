import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist
from pyro.nn import pyro_method

from ...settings import settings
from .base import PyroLikelihoodWithDispersion


class PyroNormal(PyroLikelihoodWithDispersion):
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
        return dist.Normal(estimate, torch.reciprocal(dispersion + settings.get("eps")))
