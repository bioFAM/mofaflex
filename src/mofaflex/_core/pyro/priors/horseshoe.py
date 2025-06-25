from collections.abc import Mapping, Sequence
from typing import Literal

import pyro
import pyro.distributions as dist
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ...utils import MeanStd
from ..utils import PyroParameterDict
from .base import Prior


class Horseshoe(Prior):
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
        regularized: bool = True,
        prior_scales: Mapping[str, NDArray] | None = None,
        **kwargs,
    ):
        super().__init__(names, factor_dim, nonfactor_dim, n_factors, n_nonfactors)

        if prior_scales is not None and len(prior_scales):
            regularized = True
            for name in names:
                if name in prior_scales:
                    self.register_buffer(f"_prior_scales_{name}", torch.as_tensor(prior_scales[name]))

        self._regularized = regularized

        self._global_scale_locs = PyroParameterDict()
        self._inter_scale_locs = PyroParameterDict()
        self._local_scale_locs = PyroParameterDict()
        self._caux_locs = PyroParameterDict()
        self._locs = PyroParameterDict()

        self._global_scale_scales = PyroParameterDict()
        self._inter_scale_scales = PyroParameterDict()
        self._local_scale_scales = PyroParameterDict()
        self._caux_scales = PyroParameterDict()
        self._scales = PyroParameterDict()

        ndims = abs(min(factor_dim, nonfactor_dim))
        inter_scale_shape = [1] * ndims
        inter_scale_shape[factor_dim] = n_factors

        for name in self._names:
            self._global_scale_locs[name] = PyroParam(torch.full((1,), init_loc))
            self._global_scale_scales[name] = PyroParam(
                torch.full((1,), init_scale), constraint=constraints.softplus_positive
            )
            self._inter_scale_locs[name] = PyroParam(torch.full(inter_scale_shape, init_loc))
            self._inter_scale_scales[name] = PyroParam(
                torch.full(inter_scale_shape, init_scale), constraint=constraints.softplus_positive
            )
            self._local_scale_locs[name] = PyroParam(torch.full(self._shapes[name], init_loc))
            self._local_scale_scales[name] = PyroParam(
                torch.full(self._shapes[name], init_scale), constraint=constraints.softplus_positive
            )
            self._caux_locs[name] = PyroParam(torch.full(self._shapes[name], init_loc))
            self._caux_scales[name] = PyroParam(
                torch.full(self._shapes[name], init_scale), constraint=constraints.softplus_positive
            )

            if init_tensor is not None:
                loc = init_tensor[name]["loc"]
                scale = init_tensor[name]["scale"]
            else:
                loc = torch.full(self._shapes[name], init_loc)
                scale = torch.full(self._shapes[name], init_scale)
            self._locs[name] = PyroParam(loc)
            self._scales[name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def _get_prior_scale(self, name: str):
        return getattr(self, f"_prior_scales_{name}", None)

    def _model(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        global_scale = pyro.sample(f"global_scale_z_{name}", dist.HalfCauchy(torch.ones((1,))))
        with factor_plate:
            inter_scale = pyro.sample(f"inter_scale_z_{name}", dist.HalfCauchy(torch.ones((1,))))
            with nonfactor_plate:
                local_scale = pyro.sample(f"local_scale_z_{name}", dist.HalfCauchy(torch.ones((1,))))
                local_scale = local_scale * inter_scale * global_scale
                if self._regularized:
                    caux = pyro.sample(
                        f"caux_z_{name}", dist.InverseGamma(torch.full((1,), 0.5), torch.full((1,), 0.5))
                    )
                    c = torch.sqrt(caux)
                    if (prior_scale := self._get_prior_scale(name)) is not None:
                        c = c * prior_scale.reshape(self._shapes[name])
                    local_scale = (c * local_scale) / torch.sqrt(c**2 + local_scale**2)
                return pyro.sample(f"z_{name}", dist.Normal(torch.zeros((1,)), local_scale))

    def _guide(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        pyro.sample(
            f"global_scale_z_{name}", dist.LogNormal(self._global_scale_locs[name], self._global_scale_scales[name])
        )
        with factor_plate:
            pyro.sample(
                f"inter_scale_z_{name}", dist.LogNormal(self._inter_scale_locs[name], self._inter_scale_scales[name])
            )
            with nonfactor_plate as index:
                local_scale_loc = self._local_scale_locs[name].index_select(nonfactor_plate.dim, index)
                local_scale_scale = self._local_scale_scales[name].index_select(nonfactor_plate.dim, index)
                pyro.sample(f"local_scale_z_{name}", dist.LogNormal(local_scale_loc, local_scale_scale))

                if self._regularized:
                    caux_loc = self._caux_locs[name].index_select(nonfactor_plate.dim, index)
                    caux_scale = self._caux_scales[name].index_select(nonfactor_plate.dim, index)
                    pyro.sample(f"caux_z_{name}", dist.LogNormal(caux_loc, caux_scale))

                return pyro.sample(
                    f"z_{name}",
                    dist.Normal(
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
