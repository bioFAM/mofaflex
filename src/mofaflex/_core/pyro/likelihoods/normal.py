from collections.abc import Mapping

import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist
from pyro.nn import pyro_method

from ...settings import settings
from .base import PyroLikelihoodWithDispersion, PyroLikelihoodWithShiftMixin


class PyroNormal(PyroLikelihoodWithShiftMixin, PyroLikelihoodWithDispersion):
    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        nsamples: Mapping[str, int],
        nfeatures: int,
        shift: Mapping[str, NDArray[np.floating]],
        scale: np.floating | Mapping[str, np.floating],
        *,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
    ):
        super().__init__(
            view_name,
            sample_dim,
            feature_dim,
            nsamples,
            nfeatures,
            init_loc=init_loc,
            init_scale=init_scale,
            shift=shift,
        )

        if isinstance(scale, dict):
            for group_name, gscale in scale.items():
                self.register_buffer(f"_scale_{group_name}", torch.as_tensor(gscale))
        else:
            self.register_buffer("_normal_scale", torch.as_tensor(scale))

    def _get_scale(self, group_name: str):
        try:
            return self._normal_scale
        except AttributeError:
            return getattr(self, f"_scale_{group_name}")

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
        return dist.Normal(
            estimate + self._get_shift(group_name),
            torch.reciprocal(dispersion * self._get_scale(group_name) + settings.get("eps")),
        )
