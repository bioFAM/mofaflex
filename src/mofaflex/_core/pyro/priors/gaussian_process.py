from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Literal

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.nn import PyroParam, pyro_method

from ...utils import MeanStd
from .base import Prior

if TYPE_CHECKING:
    from ...priors.gaussian_process import GP


class GaussianProcess(Prior):
    def __init__(
        self,
        names: Sequence[str],
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        gp: "GP",
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

        shape = [1] * ndims
        shape[min(factor_dim, nonfactor_dim)] = n_factors
        shape[max(factor_dim, nonfactor_dim)] = -1
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

    def _get_nonfactor_plate(self, nonfactor_plates: Mapping[str, pyro.plate]) -> pyro.plate:
        """Make combined sample plate."""
        offset = 0
        subsample = []
        nonfactor_dim = None
        for name in self._names:
            splate = nonfactor_plates[name]
            subsample.append(splate.indices + offset)
            offset += splate.size
            nonfactor_dim = splate.dim
        subsample = torch.cat(subsample)
        return pyro.plate("gp_nonfactors", offset, dim=nonfactor_dim, subsample=subsample)

    @pyro_method
    def model(
        self,
        factor_plate: pyro.plate,
        nonfactor_plates: Mapping[str, pyro.plate],
        gp_covariates: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        gnames = list(filter(lambda x: x in gp_covariates, self._names))
        covars = torch.cat(tuple(gp_covariates[g] for g in gnames), dim=0)
        idx = torch.cat(tuple(self._get_idx(g).expand(gp_covariates[g].shape[0]) for g in gnames), dim=0)
        f_dist = self._gp.pyro_model((idx[..., None], covars), name_prefix="gp")

        nonfactor_plate = self._get_nonfactor_plate(nonfactor_plates)
        with pyro.plate("gp_batch", factor_plate.size, dim=-2):  # needs to be dim=-2 to work with GPyTorch
            f = pyro.sample("gp.f", f_dist)
        new_f_shape = list(f.shape)
        new_f_shape[-len(self._full_gp_shape) :] = self._full_gp_shape
        f = f.reshape(new_f_shape)
        if factor_plate.dim > nonfactor_plate.dim:
            f = f.swapaxes(factor_plate.dim, nonfactor_plate.dim)

        outputscale = self._gp.outputscale.reshape(self._gp_shape)

        with factor_plate, nonfactor_plate:
            return dict(
                zip(
                    self._names,
                    torch.split(
                        pyro.sample("z", dist.Normal(f, 1 - outputscale)),
                        tuple(gp_covariates[g].shape[0] for g in gnames),
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
        gp_covariates: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        gnames = list(filter(lambda x: x in gp_covariates, self._names))
        covars = torch.cat(tuple(gp_covariates[g] for g in gnames), dim=0)
        idx = torch.cat(tuple(self._get_idx(g).expand(gp_covariates[g].shape[0]) for g in gnames), dim=0)
        f_dist = self._gp.pyro_guide((idx[..., None], covars), name_prefix="gp")

        nonfactor_plate = self._get_nonfactor_plate(nonfactor_plates)
        with pyro.plate("gp_batch", factor_plate.size, dim=-2):  # needs to be dim=-2 to work with GPyTorch
            pyro.sample("gp.f", f_dist)

        with factor_plate, nonfactor_plate as index:
            return dict(
                zip(
                    self._names,
                    torch.split(
                        pyro.sample(
                            "z",
                            dist.Normal(
                                self._loc.index_select(nonfactor_plate.dim, index),
                                self._scale.index_select(nonfactor_plate.dim, index),
                            ),
                        ),
                        tuple(gp_covariates[g].shape[0] for g in gnames),
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
