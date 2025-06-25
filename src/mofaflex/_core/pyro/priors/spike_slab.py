from collections.abc import Mapping, Sequence
from typing import Literal

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ...settings import settings
from ...utils import MeanStd, ShapeRate
from ..dist import ReinMaxBernoulli
from ..utils import PyroParameterDict
from .base import Prior


class SnS(Prior):
    _factors = True
    _weights = True

    def __init__(
        self,
        names: Sequence[str],
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], torch.Tensor]] | None = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        init_shape: float = 10.0,
        init_rate: float = 10.0,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        init_prob: float = 0.5,
        **kwargs,
    ):
        super().__init__(names, factor_dim, nonfactor_dim, n_factors, n_nonfactors)

        self.__shapes = PyroParameterDict()
        self._rates = PyroParameterDict()
        self._alphas = PyroParameterDict()
        self._betas = PyroParameterDict()
        self._probs = PyroParameterDict()
        self._locs = PyroParameterDict()
        self._scales = PyroParameterDict()

        ndims = abs(min(factor_dim, nonfactor_dim))
        shape = [1] * ndims
        shape[factor_dim] = n_factors

        for name in self._names:
            self.__shapes[name] = PyroParam(torch.full(shape, init_shape), constraint=constraints.softplus_positive)
            self._rates[name] = PyroParam(torch.full(shape, init_rate), constraint=constraints.softplus_positive)
            self._alphas[name] = PyroParam(torch.full(shape, init_alpha), constraint=constraints.softplus_positive)
            self._betas[name] = PyroParam(torch.full(shape, init_beta), constraint=constraints.softplus_positive)
            self._probs[name] = PyroParam(
                torch.full(self._shapes[name], init_prob), constraint=constraints.unit_interval
            )

            if init_tensor is not None:
                loc = init_tensor[name]["loc"]
                scale = init_tensor[name]["scale"]
            else:
                loc = torch.full(self._shapes[name], init_loc)
                scale = torch.full(self._shapes[name], init_scale)
            self._locs[name] = PyroParam(loc)
            self._scales[name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def _model(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate:
            alpha = pyro.sample(f"alpha_z_{name}", dist.Gamma(torch.full((1,), 1e-3), torch.full((1,), 1e-3)))
            theta = pyro.sample(f"theta_z_{name}", dist.Beta(torch.ones((1,)), torch.ones((1,))))
            with nonfactor_plate:
                s = pyro.sample(f"s_z_{name}", dist.Bernoulli(theta))
                return pyro.sample(f"z_{name}", dist.Normal(torch.zeros((1,)), 1.0 / (alpha + settings.get("eps")))) * s

    def _guide(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate:
            pyro.sample(f"alpha_z_{name}", dist.Gamma(self.__shapes[name], self._rates[name]))
            pyro.sample(f"theta_z_{name}", dist.Beta(self._alphas[name], self._betas[name]))
            with nonfactor_plate as index:
                pyro.sample(
                    f"s_z_{name}",
                    ReinMaxBernoulli(temperature=2.0, probs=self._probs[name].index_select(nonfactor_plate.dim, index)),
                )

                return pyro.sample(
                    f"z_{name}",
                    dist.Normal(
                        self._locs[name].index_select(nonfactor_plate.dim, index),
                        self._scales[name].index_select(nonfactor_plate.dim, index),
                    ),
                )

    @property
    def learning_rate_multipliers(self) -> dict[str, float]:
        return {name: 10.0 for name, _ in self._probs.named_pyro_params(prefix="_probs")}

    @property
    def posterior(self) -> MeanStd:
        posteriors = MeanStd({}, {})
        for name in self._names:
            posteriors.mean[name] = self._locs[name].squeeze(self._squeezedims)
            posteriors.std[name] = self._scales[name].squeeze(self._squeezedims)
        return posteriors

    @property
    def posterior_precision(self) -> ShapeRate:
        posteriors = ShapeRate({}, {})
        for name in self._names:
            posteriors.shape[name] = self.__shapes[name].squeeze(self._squeezedims)
            posteriors.rate[name] = self._rates[name].squeeze(self._squeezedims)
        return posteriors

    @property
    def posterior_probability(self) -> dict[str, torch.Tensor]:
        return {name: self._probs[name].squeeze(self._squeezedims) for name in self._names}
