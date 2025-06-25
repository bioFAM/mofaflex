from collections.abc import Mapping, Sequence

import pyro
import pyro.distributions as dist
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam, pyro_method

from ...gp import GP
from ...utils import MeanStd
from .base import Factor


class GP(Factor):
    def __init__(
        self,
        group_names: Sequence[str],
        factor_dim: int,
        sample_dim: int,
        n_factors: int,
        n_samples: Mapping[str, int],
        gp: GP,
        init_tensor: Mapping[str, Mapping[str, NDArray]] | None = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(group_names, factor_dim, sample_dim, n_factors, n_samples)

        self._gp = pyro.module("gp", gp)
        self._group_sizes = [n_samples[g] for g in self._group_names]
        self._sample_dim = sample_dim
        for i, g in enumerate(self._group_names):
            self.register_buffer(f"_group_idx_{g}", torch.as_tensor(i))

        ndims = abs(min(factor_dim, sample_dim))
        shape = [1] * ndims
        shape[factor_dim] = n_factors
        self._gp_shape = tuple(shape)
        n_gp_samples = sum(n_samples[g] for g in self._group_names)
        shape[sample_dim] = n_gp_samples
        self._full_gp_shape = tuple(shape)

        if init_tensor is not None:
            loc = torch.concatenate([init_tensor[group_name]["loc"] for group_name in self._group_names])
            scale = torch.concatenate([init_tensor[group_name]["scale"] for group_name in self._group_names])
        else:
            loc = torch.full(shape, init_loc)
            scale = torch.full(shape, init_scale)
        self._loc = PyroParam(loc)
        self._scale = PyroParam(scale, constraint=constraints.softplus_positive)

    def _get_group_idx(self, group_name: str):
        return getattr(self, f"_group_idx_{group_name}")

    @pyro_method
    def model(
        self,
        factor_plate: pyro.plate,
        sample_plates: Mapping[str, pyro.plate],
        covariates: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # Inducing values p(u)
        prior_distribution = self._gp.variational_strategy.prior_distribution
        prior_distribution = prior_distribution.to_event(len(prior_distribution.batch_shape))
        pyro.sample("gp.u", prior_distribution)

        # Draw samples from p(f)
        gnames = list(filter(lambda x: x in covariates, self._group_names))
        covars = torch.cat(tuple(covariates[g] for g in gnames), dim=0)
        group_idx = torch.cat(tuple(self._get_group_idx(g).expand(covariates[g].shape[0]) for g in gnames), dim=0)
        f_dist = self._gp(group_idx[..., None], covars, prior=True)
        f_dist = dist.Normal(loc=f_dist.mean, scale=f_dist.stddev).to_event(len(f_dist.event_shape) - 1)

        with pyro.plate("gp_batch", factor_plate.size, dim=-2):  # needs to be dim=-2 to work with GPyTorch
            f = pyro.sample("gp.f", f_dist.mask(False)).reshape(self._full_gp_shape)

        outputscale = self._gp.outputscale.reshape(self._gp_shape)

        with factor_plate:
            return dict(
                zip(
                    self._group_names,
                    torch.split(
                        pyro.sample("z", dist.Normal(f, 1 - outputscale)),
                        tuple(covariates.get(g, torch.as_tensor(())).shape[0] for g in self._group_names),
                        dim=self._sample_dim,
                    ),
                    strict=False,
                )
            )

    @pyro_method
    def guide(
        self,
        factor_plate: pyro.plate,
        sample_plates: Mapping[str, pyro.plate],
        covariates: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # make combined sample plate
        offset = 0
        subsample = []
        sample_dim = None
        for group_name in self._group_names:
            splate = sample_plates[group_name]
            subsample.append(splate.indices + offset)
            offset += splate.size
            sample_dim = splate.dim
        subsample = torch.cat(subsample)
        gp_sample_plate = pyro.plate("gp_samples", offset, dim=sample_dim, subsample=subsample)

        # Inducing values q(u)
        variational_distribution = self._gp.variational_strategy.variational_distribution
        variational_distribution = variational_distribution.to_event(len(variational_distribution.batch_shape))
        pyro.sample("gp.u", variational_distribution)

        gnames = list(filter(lambda x: x in covariates, self._group_names))
        covars = torch.cat(tuple(covariates[g] for g in gnames), dim=0)
        group_idx = torch.cat(tuple(self._get_group_idx(g).expand(covariates[g].shape[0]) for g in gnames), dim=0)
        with pyro.plate("gp_batch", factor_plate.size, dim=-2):  # needs to be dim=-2 to work with GPyTorch
            # Draw samples from q(f)
            f_dist = self._gp(group_idx[..., None], covars, prior=False)
            f_dist = dist.Normal(f_dist.mean, f_dist.stddev).to_event(len(f_dist.event_shape) - 1)
            pyro.sample("gp.f", f_dist.mask(False))

        with factor_plate, gp_sample_plate as index:
            return dict(
                zip(
                    self._group_names,
                    torch.split(
                        pyro.sample(
                            "z",
                            dist.Normal(
                                self._loc.index_select(gp_sample_plate.dim, index),
                                self._scale.index_select(gp_sample_plate.dim, index),
                            ),
                        ),
                        tuple(covariates.get(g, torch.as_tensor(())).shape[0] for g in self._group_names),
                        dim=self._sample_dim,
                    ),
                    strict=False,
                )
            )

    @property
    def posterior(self) -> MeanStd:
        loc = dict(
            zip(self._group_names, torch.split(self._loc, self._group_sizes, dim=self._sample_dim), strict=False)
        )
        scale = dict(
            zip(self._group_names, torch.split(self._scale, self._group_sizes, dim=self._sample_dim), strict=False)
        )
        factors = MeanStd(loc, scale)
        for res in factors:
            for k, v in res.items():
                res[k] = v.squeeze(self._squeezedims)
        return factors
