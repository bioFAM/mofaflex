from collections.abc import Mapping, Sequence
from typing import Literal

import pyro
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ..utils import MeanStd, PyroParameterDict
from .base import Prior


class _SimpleLocationScale(Prior):
    def __init__(self, names: Sequence[str], prior_dist: type[pyro.distributions.Distribution]):
        super().__init__(names)

        self._prior_dist = prior_dist

    def on_train_start(
        self,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] | None = None,
    ):
        init_loc: float = 0.0
        init_scale: float = 0.1

        self._locs = PyroParameterDict()
        self._scales = PyroParameterDict()

        for name in self._names:
            if init_tensor is not None:
                loc = init_tensor[name]["loc"]
                scale = init_tensor[name]["scale"]
            else:
                loc = torch.full((n_nonfactors[name], n_factors), init_loc)
                scale = torch.full((n_nonfactors[name], n_factors), init_scale)
            self._locs[name] = PyroParam(loc)
            self._scales[name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def _model(
        self, id: str, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs
    ) -> torch.Tensor:
        with factor_plate, nonfactor_plate:
            return pyro.sample(f"{id}_z_{name}", self._prior_dist(torch.zeros((1,)), torch.ones((1,))))

    def _guide(
        self, id: str, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs
    ) -> torch.Tensor:
        with factor_plate, nonfactor_plate as index:
            return pyro.sample(
                f"{id}_z_{name}", pyro.distributions.Normal(self._locs[name][index, :], self._scales[name][index, :])
            )

    @property
    def posterior(self) -> MeanStd:
        posteriors = MeanStd({}, {})
        for name in self._names:
            posteriors.mean[name] = self._locs[name]
            posteriors.std[name] = self._scales[name]
        return posteriors


class Normal(_SimpleLocationScale):
    """Standard Normal prior."""

    def __init__(self, names: Sequence[str]):
        super().__init__(names, pyro.distributions.Normal)


class Laplace(_SimpleLocationScale):
    """Standard Laplace prior."""

    def __init__(self, names: Sequence[str]):
        super().__init__(names, pyro.distributions.Laplace)
