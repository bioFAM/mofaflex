from collections.abc import Mapping, Sequence
from typing import Literal

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.nn import PyroParam, pyro_method

from ...gp import GP
from ...utils import MeanStd
from .base import Prior


class GP(Prior):
    _factors = True
    _weights = False

    def __init__(
        self,
        names: Sequence[str],
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        gp: GP,
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], torch.Tensor]] | None = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(names, factor_dim, nonfactor_dim, n_factors, n_nonfactors)

        self._gp = pyro.module("gp", gp)
        self._sizes = [n_nonfactors[g] for g in self._names]
        self._nonfactor_dim = nonfactor_dim
        for i, g in enumerate(self._names):
            self.register_buffer(f"_idx_{g}", torch.as_tensor(i))

        ndims = abs(min(factor_dim, nonfactor_dim))
        shape = [1] * ndims
        shape[factor_dim] = n_factors
        self._gp_shape = tuple(shape)
        n_gp_samples = sum(n_nonfactors[g] for g in self._names)
        shape[nonfactor_dim] = n_gp_samples
        self._full_gp_shape = tuple(shape)

        if init_tensor is not None:
            loc = torch.concatenate([init_tensor[name]["loc"] for name in self._names], dim=nonfactor_dim)
            scale = torch.concatenate([init_tensor[name]["scale"] for name in self._names], dim=nonfactor_dim)
        else:
            loc = torch.full(shape, init_loc)
            scale = torch.full(shape, init_scale)
        self._loc = PyroParam(loc)
        self._scale = PyroParam(scale, constraint=constraints.softplus_positive)

    def _get_idx(self, group_name: str):
        return getattr(self, f"_idx_{group_name}")

    @pyro_method
    def model(
        self,
        factor_plate: pyro.plate,
        nonfactor_plates: Mapping[str, pyro.plate],
        covariates: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Inducing values p(u)
        prior_distribution = self._gp.variational_strategy.prior_distribution
        prior_distribution = prior_distribution.to_event(len(prior_distribution.batch_shape))
        pyro.sample("gp.u", prior_distribution)

        # Draw samples from p(f)
        gnames = list(filter(lambda x: x in covariates, self._names))
        covars = torch.cat(tuple(covariates[g] for g in gnames), dim=0)
        idx = torch.cat(tuple(self._get_idx(g).expand(covariates[g].shape[0]) for g in gnames), dim=0)
        f_dist = self._gp(idx[..., None], covars, prior=True)
        f_dist = dist.Normal(loc=f_dist.mean, scale=f_dist.stddev).to_event(len(f_dist.event_shape) - 1)

        with pyro.plate("gp_batch", factor_plate.size, dim=-2):  # needs to be dim=-2 to work with GPyTorch
            f = pyro.sample("gp.f", f_dist.mask(False)).reshape(self._full_gp_shape)

        outputscale = self._gp.outputscale.reshape(self._gp_shape)

        with factor_plate:
            return dict(
                zip(
                    self._names,
                    torch.split(
                        pyro.sample("z", dist.Normal(f, 1 - outputscale)),
                        tuple(covariates[g].shape[0] for g in gnames),
                        dim=self._nonfactor_dim,
                    ),
                    strict=False,
                )
            )

    @pyro_method
    def guide(
        self,
        factor_plate: pyro.plate,
        nonfactor_plates: Mapping[str, pyro.plate],
        covariates: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # make combined sample plate
        offset = 0
        subsample = []
        nonfactor_dim = None
        for name in self._names:
            splate = nonfactor_plates[name]
            subsample.append(splate.indices + offset)
            offset += splate.size
            nonfactor_dim = splate.dim
        subsample = torch.cat(subsample)
        gp_nonfactor_plate = pyro.plate("gp_nonfactors", offset, dim=nonfactor_dim, subsample=subsample)

        # Inducing values q(u)
        variational_distribution = self._gp.variational_strategy.variational_distribution
        variational_distribution = variational_distribution.to_event(len(variational_distribution.batch_shape))
        pyro.sample("gp.u", variational_distribution)

        gnames = list(filter(lambda x: x in covariates, self._names))
        covars = torch.cat(tuple(covariates[g] for g in gnames), dim=0)
        idx = torch.cat(tuple(self._get_idx(g).expand(covariates[g].shape[0]) for g in gnames), dim=0)
        with pyro.plate("gp_batch", factor_plate.size, dim=-2):  # needs to be dim=-2 to work with GPyTorch
            # Draw samples from q(f)
            f_dist = self._gp(idx[..., None], covars, prior=False)
            f_dist = dist.Normal(f_dist.mean, f_dist.stddev).to_event(len(f_dist.event_shape) - 1)
            pyro.sample("gp.f", f_dist.mask(False))

        with factor_plate, gp_nonfactor_plate as index:
            return dict(
                zip(
                    self._names,
                    torch.split(
                        pyro.sample(
                            "z",
                            dist.Normal(
                                self._loc.index_select(gp_nonfactor_plate.dim, index),
                                self._scale.index_select(gp_nonfactor_plate.dim, index),
                            ),
                        ),
                        tuple(covariates[g].shape[0] for g in gnames),
                        dim=self._nonfactor_dim,
                    ),
                    strict=False,
                )
            )

    @property
    def posterior(self) -> MeanStd:
        loc = dict(zip(self._names, torch.split(self._loc, self._sizes, dim=self._nonfactor_dim), strict=False))
        scale = dict(zip(self._names, torch.split(self._scale, self._sizes, dim=self._nonfactor_dim), strict=False))
        posteriors = MeanStd(loc, scale)
        for res in posteriors:
            for k, v in res.items():
                res[k] = v.squeeze(self._squeezedims)
        return posteriors
