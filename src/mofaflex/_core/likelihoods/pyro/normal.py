from collections.abc import Mapping

import numpy as np
import pyro
import torch
from numpy.typing import NDArray
from pyro import distributions as dist
from pyro.nn import pyro_method

from ...settings import settings
from .base import LikelihoodWithDispersion


class Normal(LikelihoodWithDispersion):
    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        nsamples: Mapping[str, int],
        nfeatures: int,
        *,
        shift: Mapping[str, NDArray[np.floating]] | None = None,
        scale: np.floating | Mapping[str, np.floating] | None = None,
        init_loc: float = 1.0,
        init_scale: float = 0.1,
    ):
        if scale is not None:
            try:
                init_loc = np.mean(list(scale.values()))
                scale = {group_name: torch.as_tensor(gscale) for group_name, gscale in scale.items()}
            except AttributeError:
                init_loc = scale
                scale = torch.as_tensor(scale)
        else:
            scale = None

        super().__init__(
            view_name, sample_dim, feature_dim, nsamples, nfeatures, init_loc=init_loc, init_scale=init_scale
        )
        self._shift = (
            {group_name: torch.as_tensor(gshift) for group_name, gshift in shift.items()}
            if shift is not None
            else shift
        )
        self.__scale = scale

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
        if self._shift is not None and self.__scale is not None:
            try:
                scale = self.__scale[group_name]
            except IndexError:
                scale = self.__scale
            estimate = estimate * scale + self._shift[group_name][feature_plate.indices[nonmissing_features]]
        return dist.Normal(estimate, dispersion + settings.get("eps"))
