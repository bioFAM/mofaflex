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

    def _on_train_start(
        self,
        factor_dim: int,
        nonfactor_dim: int,
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
                loc = torch.full(self._shapes[name], init_loc)
                scale = torch.full(self._shapes[name], init_scale)
            self._locs[name] = PyroParam(loc)
            self._scales[name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def _model(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate, nonfactor_plate:
            return pyro.sample(f"z_{name}", self._prior_dist(torch.zeros((1,)), torch.ones((1,))))

    def _guide(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate, nonfactor_plate as index:
            return pyro.sample(
                f"z_{name}",
                pyro.distributions.Normal(
                    self._locs[name].index_select(nonfactor_plate.dim, index),
                    self._scales[name].index_select(nonfactor_plate.dim, index),
                ),
            )

    @property
    def posterior(self) -> MeanStd:
        posteriors = MeanStd({}, {})
        for name in self._names:
            posteriors.mean[name] = self._locs[name].squeeze(self._squeezedims)
            posteriors.std[name] = self._scales[name].squeeze(self._squeezedims)
        return posteriors


class Normal(_SimpleLocationScale):
    """Standard Normal prior."""

    def __init__(self, names: Sequence[str]):
        super().__init__(names, pyro.distributions.Normal)


class Laplace(_SimpleLocationScale):
    """Standard Laplace prior."""

    def __init__(self, names: Sequence[str]):
        super().__init__(names, pyro.distributions.Laplace)
