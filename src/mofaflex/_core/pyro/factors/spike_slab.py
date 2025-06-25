from collections.abc import Mapping, Sequence

import pyro
import pyro.distributions as dist
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ... import settings
from ...utils import MeanStd, ShapeRate
from ..dist import ReinMaxBernoulli
from .base import Factor, PyroParameterDict


class SnS(Factor):
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
        init_shape: float = 10.0,
        init_rate: float = 10.0,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        init_prob: float = 0.5,
        regularized: bool = True,
        **kwargs,
    ):
        super().__init__(group_names, factor_dim, sample_dim, n_factors, n_samples)

        self._regularized = regularized

        self._shapes = PyroParameterDict()
        self._rates = PyroParameterDict()
        self._alphas = PyroParameterDict()
        self._betas = PyroParameterDict()
        self._probs = PyroParameterDict()
        self._locs = PyroParameterDict()
        self._scales = PyroParameterDict()

        ndims = abs(min(factor_dim, sample_dim))
        shape = [1] * ndims
        shape[factor_dim] = n_factors

        for group_name in self._group_names:
            self._shapes[group_name] = PyroParam(
                torch.full(shape, init_shape), constraint=constraints.softplus_positive
            )
            self._rates[group_name] = PyroParam(torch.full(shape, init_rate), constraint=constraints.softplus_positive)
            self._alphas[group_name] = PyroParam(
                torch.full(shape, init_alpha), constraint=constraints.softplus_positive
            )
            self._betas[group_name] = PyroParam(torch.full(shape, init_beta), constraint=constraints.softplus_positive)
            self._probs[group_name] = PyroParam(
                torch.full(self._shapes[group_name], init_prob), constraint=constraints.unit_interval
            )

            if init_tensor is not None:
                loc = torch.as_tensor(init_tensor[group_name]["loc"])
                scale = torch.as_tensor(init_tensor[group_name]["scale"])
            else:
                loc = torch.full(self._shapes[group_name], init_loc)
                scale = torch.full(self._shapes[group_name], init_scale)
            self._locs[group_name] = PyroParam(loc)
            self._scales[group_name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def _model(self, group_name: str, factor_plate: pyro.plate, sample_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate:
            alpha = pyro.sample(f"alpha_z_{group_name}", dist.Gamma(torch.full((1,), 1e-3), torch.full((1,), 1e-3)))
            theta = pyro.sample(f"theta_z_{group_name}", dist.Beta(torch.ones((1,)), torch.ones((1,))))
            with sample_plate:
                s = pyro.sample(f"s_z_{group_name}", dist.Bernoulli(theta))
                return (
                    pyro.sample(f"z_{group_name}", dist.Normal(torch.zeros((1,)), 1.0 / (alpha + settings.get("eps"))))
                    * s
                )

    def _guide(self, group_name: str, factor_plate: pyro.plate, sample_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate:
            pyro.sample(f"alpha_z_{group_name}", dist.Gamma(self._shapes[group_name], self._rates[group_name]))
            pyro.sample(f"theta_z_{group_name}", dist.Beta(self._alphas[group_name], self._betas[group_name]))
            with sample_plate as index:
                prob = self._probs[group_name]
                if index is not None:
                    prob = prob.index_select(sample_plate.dim, index)
                pyro.sample(f"s_z_{group_name}", ReinMaxBernoulli(temperature=2.0, probs=prob))

                return pyro.sample(
                    f"z_{group_name}",
                    dist.Normal(
                        self._locs[group_name].index_select(sample_plate.dim, index),
                        self._scales[group_name].index_select(sample_plate.dim, index),
                    ),
                )

    @property
    def posterior(self) -> MeanStd:
        factors = MeanStd({}, {})
        for group_name in self._group_names:
            factors.mean[group_name] = self._factor_locs[group_name].squeeze(self._squeezedims)
            factors.std[group_name] = self._factor_scales[group_name].squeeze(self._squeezedims)
        return factors

    @property
    def posterior_precision(self) -> ShapeRate:
        factors = ShapeRate({}, {})
        for group_name in self._group_names:
            factors.shape[group_name] = self._shapes[group_name].squeeze(self._squeezedims)
            factors.rate[group_name] = self._rates[group_name].squeeze(self._squeezedims)
        return factors

    @property
    def posterior_probability(self) -> dict[str, torch.Tensor]:
        return {group_name: self._probs[group_name].squeeze(self._squeezedims) for group_name in self._group_names}
