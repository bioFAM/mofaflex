from collections.abc import Mapping, Sequence

import pyro
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ...utils import MeanStd
from .base import Factor, PyroParameterDict


class _SimpleLocationScale(Factor):
    def __init__(
        self,
        prior_dist: type[pyro.distributions.Distribution],
        group_names: Sequence[str],
        factor_dim: int,
        sample_dim: int,
        n_factors: int,
        n_samples: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[str, NDArray]] | None = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
    ):
        super().__init__(group_names, factor_dim, sample_dim, n_factors, n_samples)

        self._prior_dist = prior_dist
        self._locs = PyroParameterDict()
        self._scales = PyroParameterDict()

        for group_name in self._group_names:
            if init_tensor is not None:
                loc = torch.as_tensor(init_tensor[group_name]["loc"])
                scale = torch.as_tensor(init_tensor[group_name]["scale"])
            else:
                loc = torch.full(self._shapes[group_name], init_loc)
                scale = torch.full(self._shapes[group_name], init_scale)
            self._locs[group_name] = PyroParam(loc)
            self._scales[group_name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def _model(self, group_name: str, factor_plate: pyro.plate, sample_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate, sample_plate:
            return pyro.sample(f"z_{group_name}", self._prior_dist(torch.zeros((1,)), torch.ones((1,))))

    def _guide(self, group_name: str, factor_plate: pyro.plate, sample_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate, sample_plate as index:
            return pyro.sample(
                f"z_{group_name}",
                pyro.distributions.Normal(
                    self._locs[group_name].index_select(sample_plate.dim, index),
                    self._scales[group_name].index_select(sample_plate.dim, index),
                ),
            )

    @property
    def posterior(self) -> MeanStd:
        factors = MeanStd({}, {})
        for group_name in self._group_names:
            factors.mean[group_name] = self._locs[group_name].squeeze(self._squeezedims)
            factors.std[group_name] = self._scales[group_name].squeeze(self._squeezedims)
        return factors


class Normal(_SimpleLocationScale):
    def __init__(
        self,
        group_names: Sequence[str],
        factor_dim: int,
        sample_dim: int,
        n_factors: int,
        n_samples: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[str, NDArray]] | None = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            pyro.distributions.Normal,
            group_names,
            factor_dim,
            sample_dim,
            n_factors,
            n_samples,
            init_tensor,
            init_loc,
            init_scale,
        )


class Laplace(_SimpleLocationScale):
    def __init__(
        self,
        group_names: Sequence[str],
        factor_dim: int,
        sample_dim: int,
        n_factors: int,
        n_samples: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[str, NDArray]] | None = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            pyro.distributions.Laplace,
            group_names,
            factor_dim,
            sample_dim,
            n_factors,
            n_samples,
            init_tensor,
            init_loc,
            init_scale,
        )
