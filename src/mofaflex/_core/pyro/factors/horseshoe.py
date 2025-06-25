from collections.abc import Mapping, Sequence

import pyro
import pyro.distributions as dist
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ...utils import MeanStd
from .base import Factor, PyroParameterDict


class HorseShoe(Factor):
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
        regularized=True,
        **kwargs,
    ):
        super().__init__(group_names, factor_dim, sample_dim, n_factors, n_samples)

        self._regularized = regularized

        self._global_scale_locs = PyroParameterDict()
        self._inter_scale_locs = PyroParameterDict()
        self._local_scale_locs = PyroParameterDict()
        self._caux_locs = PyroParameterDict()
        self._factor_locs = PyroParameterDict()

        self._global_scale_scales = PyroParameterDict()
        self._inter_scale_scales = PyroParameterDict()
        self._local_scale_scales = PyroParameterDict()
        self._caux_scales = PyroParameterDict()
        self._factor_scales = PyroParameterDict()

        ndims = abs(min(factor_dim, sample_dim))
        inter_scale_shape = [1] * ndims
        inter_scale_shape[factor_dim] = n_factors

        for group_name in self._group_names:
            self._global_scale_locs[group_name] = PyroParam(torch.full((1,), init_loc))
            self._global_scale_scales[group_name] = PyroParam(
                torch.full((1,), init_scale), constraint=constraints.softplus_positive
            )
            self._inter_scale_locs[group_name] = PyroParam(torch.full(inter_scale_shape, init_loc))
            self._inter_scale_scales[group_name] = PyroParam(
                torch.full(inter_scale_shape, init_scale), constraint=constraints.softplus_positive
            )
            self._local_scale_locs[group_name] = PyroParam(torch.full(self._shapes[group_name], init_loc))
            self._local_scale_scales[group_name] = PyroParam(
                torch.full(self._shapes[group_name], init_scale), constraint=constraints.softplus_positive
            )
            self._caux_locs[group_name] = PyroParam(torch.full(self._shapes[group_name], init_loc))
            self._caux_scales[group_name] = PyroParam(
                torch.full(self._shapes[group_name], init_scale), constraint=constraints.softplus_positive
            )

            if init_tensor is not None:
                loc = torch.as_tensor(init_tensor[group_name]["loc"])
                scale = torch.as_tensor(init_tensor[group_name]["scale"])
            else:
                loc = torch.full(self._shapes[group_name], init_loc)
                scale = torch.full(self._shapes[group_name], init_scale)
            self._factor_locs[group_name] = PyroParam(loc)
            self._factor_scales[group_name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def _model(self, group_name: str, factor_plate: pyro.plate, sample_plate: pyro.plate, **kwargs) -> torch.Tensor:
        global_scale = pyro.sample(f"global_scale_z_{group_name}", dist.HalfCauchy(torch.ones((1,))))
        with factor_plate:
            inter_scale = pyro.sample(f"inter_scale_z_{group_name}", dist.HalfCauchy(torch.ones((1,))))
            with sample_plate:
                local_scale = pyro.sample(f"local_scale_z_{group_name}", dist.HalfCauchy(torch.ones((1,))))
                local_scale = local_scale * inter_scale * global_scale
                if self._regularized:
                    caux = pyro.sample(
                        f"caux_z_{group_name}", dist.InverseGamma(torch.full((1,), 0.5), torch.full((1,), 0.5))
                    )
                    c = torch.sqrt(caux)
                    local_scale = (c * local_scale) / torch.sqrt(c**2 + local_scale**2)
                return pyro.sample(f"z_{group_name}", dist.Normal(torch.zeros((1,)), local_scale))

    def _guide(self, group_name: str, factor_plate: pyro.plate, sample_plate: pyro.plate, **kwargs) -> torch.Tensor:
        pyro.sample(
            f"global_scale_z_{group_name}",
            dist.LogNormal(self._global_scale_locs[group_name], self._global_scale_scales[group_name]),
        )
        with factor_plate:
            pyro.sample(
                f"inter_scale_z_{group_name}",
                dist.LogNormal(self._inter_scale_locs[group_name], self._inter_scale_scales[group_name]),
            )
            with sample_plate as index:
                local_scale_loc = self._local_scale_locs[group_name].index_select(sample_plate.dim, index)
                local_scale_scale = self._local_scale_scales[group_name].index_select(sample_plate.dim, index)
                pyro.sample(f"local_scale_z_{group_name}", dist.LogNormal(local_scale_loc, local_scale_scale))

                if self._regularized:
                    caux_loc = self._caux_locs[group_name].index_select(sample_plate.dim, index)
                    caux_scale = self._caux_scales[group_name].index_select(sample_plate.dim, index)
                    pyro.sample(f"caux_z_{group_name}", dist.LogNormal(caux_loc, caux_scale))

                return pyro.sample(
                    f"z_{group_name}",
                    dist.Normal(
                        self._factor_locs[group_name].index_select(sample_plate.dim, index),
                        self._factor_scales[group_name].index_select(sample_plate.dim, index),
                    ),
                )

    @property
    def posterior(self) -> MeanStd:
        factors = MeanStd({}, {})
        for group_name in self._group_names:
            factors.mean[group_name] = self._factor_locs[group_name].squeeze(self._squeezedims)
            factors.std[group_name] = self._factor_scales[group_name].squeeze(self._squeezedims)
        return factors
