import logging
from collections.abc import Mapping, Sequence
from contextlib import suppress
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from dtw import dtw
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam, pyro_method

from ...datasets import MofaFlexDataset, merge_covariates
from ...utils import MeanStd, pickle_torch_state, unpickle_torch_state
from .. import Prior
from .gp import GP

_logger = logging.getLogger(__name__)


class GaussianProcess(Prior):
    """Gaussian process prior for spatially or temporally smooth factors.

    Args:
        covariates_key: The column of `.obs`/`.var` that contains covariate values. Cannot be used together with `covariates_mkey`.
        covariates_mkey: The key in `.obsm`/`.varm` that contains covariate values. Cannot be used together with `covariates_key`.
        n_inducing: Number of inducing points.
        kernel: Kernel function to use.
        mefisto_kernel: Whether to use the MEFISTO group covariance kernel or treat groups independently.
        independent_lengthscales: Whether to use a separate lengthscale per covariate dimension.
        group_cvar_rank: Rank of the group correlation matrix. Only relevant if `mefisto_kernel=True`.
        warp: Whether to use dynamic time warping. Warping is only supported for 1D covariates.
        warp_interval: Apply dynamic time warping every `warp_interval` epochs.
        warp_open_begin: Perform open-ended alignment.
        warp_open_end: Perform open-ended alignment.
        warp_reference_group: Reference group to align the others to. Defaults to the first group.
    """

    _state_attrs = (
        "_key",
        "_mkey",
        "_covariates",
        "_orig_covariates",
        "_n_inducing",
        "_kernel",
        "_mefisto_kernel",
        "_independent_lengthscales",
        "_group_covar_rank",
        "_warp",
        "_warp_interval",
        "_warp_open_begin",
        "_warp_open_end",
        "_warp_reference_group",
        "_warp_groups_order",
    )

    def __init__(
        self,
        names: str | Sequence[str],
        covariates_key: str | Mapping[str] | None = None,
        covariates_mkey: str | Mapping[str] | None = None,
        n_inducing: int = 100,
        kernel: Literal["RBF", "Matern"] = "RBF",
        mefisto_kernel: bool = True,
        independent_lengthscales: bool = False,
        group_covar_rank: int = 1,
        warp: bool = False,
        warp_interval: int = 20,
        warp_open_begin: bool = True,
        warp_open_end: bool = True,
        warp_reference_group: str | None = None,
    ):
        super().__init__(names)

        if covariates_key is None and covariates_mkey is None:
            raise ValueError("Neither `covariates_key` nor covariates_mkey` given.")
        if covariates_key is not None and covariates_mkey is not None:
            raise ValueError("Provide either `covariates_key` or `covariates_mkey`, but not both.")

        self._key = covariates_key
        self._mkey = covariates_mkey
        self._n_inducing = n_inducing
        self._kernel = kernel
        self._mefisto_kernel = mefisto_kernel
        self._independent_lengthscales = independent_lengthscales
        self._group_covar_rank = group_covar_rank
        self._warp = warp
        self._warp_interval = warp_interval
        self._warp_open_begin = warp_open_begin
        self._warp_open_end = warp_open_end
        self._warp_reference_group = warp_reference_group

        self._gp = None
        self._gps = None

    def get_datasets(
        self, data: MofaFlexDataset, axis: Literal[0, 1], n_factors: int, n_nonfactors: Mapping[str, int]
    ) -> dict[str, dict[str, pd.DataFrame]]:
        self._covariates = merge_covariates(data.get_covariates(axis, self._key, self._mkey, self._names))
        for covar in self._covariates.values():
            if pd.api.types.is_integer_dtype(covar.columns):
                covar.columns = "Covariate " + covar.columns.astype(str)
        return {"gp_covariates": self._covariates}

    def _init_gp(self, n_factors: int):
        self._gp = GP(
            n_inducing=self._n_inducing,
            covariates=(covar.to_numpy() for covar in self._covariates.values()),
            n_factors=n_factors,
            n_groups=len(self._names),
            kernel=self._kernel,
            independent_lengthscales=self._independent_lengthscales,
            rank=self._group_covar_rank,
            use_mefisto_kernel=self._mefisto_kernel,
        )

    def on_train_start(
        self,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] | None = None,
    ):
        init_loc: float = 0.0
        init_scale: float = 0.1

        if self._warp:
            if len(self.names) > 1:
                self._warp_groups_order = {}
                for g in self.names:
                    ccov = self._covariates[g].to_numpy().squeeze()
                    if ccov.ndim > 1:
                        raise ValueError(
                            f"Warping can only be performed with 1D covariates, but the covariate for group {g} has {ccov.ndim} dimensions."
                        )
                    self._warp_groups_order[g] = ccov.argsort()
                self._orig_covariates = {g: c.copy() for g, c in self._covariates.items()}

                if self._warp_reference_group is None:
                    self._warp_reference_group = self.names[0]
            elif len(self.names) == 1:
                _logger.warning("Need at least 2 groups for warping, but only one was given. Ignoring warping.")
                self._warp = False

        self._init_gp(n_factors)
        self._gp = pyro.module("gp", self._gp)

        self._sizes = [n_nonfactors[g] for g in self._names]
        self._idx = {name: torch.as_tensor(i) for i, name in enumerate(self._names)}

        if init_tensor is not None:
            loc = torch.concatenate([init_tensor[name]["loc"] for name in self._names], dim=0)
            scale = torch.concatenate([init_tensor[name]["scale"] for name in self._names], dim=0)
        else:
            n_nonfactors = sum(self._sizes)
            loc = torch.full((n_nonfactors, n_factors), init_loc)
            scale = torch.full((n_nonfactors, n_factors), init_scale)
        self._loc = PyroParam(loc)
        self._scale = PyroParam(scale, constraint=constraints.softplus_positive)

    def on_train_epoch_end(self, epoch: int):
        if self._warp and epoch > 0 and not epoch % self._warp_interval:
            factormeans = {
                group_name: mean.cpu().numpy() for group_name, mean in self.posterior.mean.items()
            }  # TODO: investigate how warping interacts with non-negativity
            reffactormeans = factormeans[self._warp_reference_group].mean(axis=1)
            refidx = self._warp_groups_order[self._warp_reference_group]
            for g in self.names:
                if g == self._warp_reference_group:
                    continue
                idx = self._warp_groups_order[g]
                alignment = dtw(
                    reffactormeans[refidx],
                    factormeans[g][idx, :].mean(axis=1),
                    open_begin=self._warp_open_begin,
                    open_end=self._warp_open_end,
                    step_pattern="asymmetric",
                )
                self._covariates[g] = self._orig_covariates[g].copy()
                self._covariates[g].iloc[idx[alignment.index2], 0] = self._orig_covariates[
                    self._warp_reference_group
                ].iloc[refidx[alignment.index1], 0]
            self._gp.update_inducing_points(covar.to_numpy() for covar in self._covariates.values())

    def on_train_end(
        self,
        data: MofaFlexDataset,
        factor_names: Sequence[str],
        nonfactor_names: Mapping[str, Sequence[str]],
        results: MeanStd,
        results_nonnegative: dict[str, bool],
        batch_size: int,
    ):
        self._gps = self._get_gps({g: covar.to_numpy() for g, covar in self._covariates.items()}, batch_size)

    @torch.inference_mode()
    def _get_gps(self, x: Mapping[str, np.ndarray | torch.Tensor], batch_size: int):
        gps = MeanStd({}, {})
        for group_idx, group_name in enumerate(self._names):
            gidx = torch.as_tensor(group_idx)
            gdata = x[group_name]
            mean, std = [], []

            for start_idx in range(0, gdata.shape[0], batch_size):
                end_idx = min(start_idx + batch_size, gdata.shape[0])
                minibatch = gdata[start_idx:end_idx]

                gp_dist = self._gp(
                    (gidx.expand(minibatch.shape[0], 1), torch.as_tensor(minibatch, dtype=torch.float32)), prior=False
                )

                mean.append(gp_dist.mean.cpu().numpy().T)
                std.append(gp_dist.stddev.cpu().numpy().T)

            gps.mean[group_name] = np.concatenate(mean, axis=0)
            gps.std[group_name] = np.concatenate(std, axis=0)
        return gps

    @Prior._api
    @property
    def a̲x̲i̲s̲_covariates_names(self) -> dict[str, NDArray[str | np.str_]]:
        """Covariate names for each group where they could be inferred from the input."""
        return {group_name: covar.columns.to_numpy() for group_name, covar in self.covariates.items()}

    @Prior._api
    @property
    def a̲x̲i̲s̲_covariates(self) -> Mapping[str, NDArray[np.float32]]:
        """Covariates for each group."""
        return (
            MappingProxyType(self._orig_covariates)
            if hasattr(self, "_orig_covariates")
            else MappingProxyType(self._covariates)
        )

    @Prior._api
    @property
    def warped_a̲x̲i̲s̲_covariates(self) -> Mapping[str, NDArray[np.float32]] | None:
        """Time-warped covariates for each group, if dynamic time warping was enabled."""
        return MappingProxyType(self._covariates) if hasattr(self, "_orig_covariates") else None

    @Prior._api
    @property
    def a̲x̲i̲s̲_gp_lengthscale(self) -> NDArray[np.float32]:
        """Inferred lengthscales for each factor."""
        return self._gp.lengthscale.detach().cpu().numpy()

    @Prior._api
    @property
    def a̲x̲i̲s̲_gp_scale(self) -> NDArray[np.float32]:
        """Inferred variance scales (smoothness) for each factor."""
        return self._gp.outputscale.detach().cpu().numpy()

    @Prior._api
    @property
    def a̲x̲i̲s̲_gp_group_correlation(self) -> NDArray[np.float32]:
        """Between-group correlation for each factor."""
        return self._gp.group_corr.detach().cpu().numpy()

    @Prior._api
    def get_a̲x̲i̲s̲_gps(
        self,
        moment: Literal["mean", "std"] = "mean",
        x: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        batch_size: int | None = None,
    ) -> Mapping[str, pd.DataFrame]:
        """Get all latent functions.

        Args:
             moment: Which moment of the posterior distribution to return.
             x: Covariate values for each group. If `None`, will return latent function values at
                covariate coordinates used for training.
             batch_size: Minibatch size. Only has an effect if `x` is not `None`. Defaults to the
                minibatch size used for training.
        """
        gp_old = getattr(self._gps, moment)
        if x is None:
            return MappingProxyType(gp_old)
        else:
            gps = getattr(self._get_gps(x, batch_size), moment)
            for group_name_calc, gp_calc in gps.items():
                gps[group_name_calc] = pd.DataFrame(gp_calc, columns=gp_old[group_name_calc].columns)
            return gps

    def _save(self) -> dict:
        state = {}
        state["gps"] = self._gps._asdict()
        if self._gp is not None:
            state["gp_state"] = pickle_torch_state(self._gp.state_dict())
        return state

    def _load(
        self, state: Mapping[str, Any], *, n_factors: int, n_nonfactors: Mapping[str, int], map_location=None, **kwargs
    ):
        self._gps = MeanStd(**state["gps"])
        self._init_gp(n_factors)
        with suppress(KeyError):
            self._gp.load_state_dict(unpickle_torch_state(state["gp_state"], map_location=map_location))

    def _get_nonfactor_plate(self, nonfactor_plates: Mapping[str, pyro.plate]) -> pyro.plate:
        """Make combined sample plate."""
        offset = 0
        subsample = []
        for name in self._names:
            splate = nonfactor_plates[name]
            subsample.append(splate.indices + offset)
            offset += splate.size
        subsample = torch.cat(subsample)
        return pyro.plate("gp_nonfactors", offset, dim=-2, subsample=subsample)

    @pyro_method
    def model(
        self,
        id: str,
        factor_plate: pyro.plate,
        nonfactor_plates: Mapping[str, pyro.plate],
        gp_covariates: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        gnames = list(filter(lambda x: x in gp_covariates, self._names))
        covars = torch.cat(tuple(gp_covariates[g] for g in gnames), dim=0)
        idx = torch.cat(tuple(self._idx[g].expand(gp_covariates[g].shape[0]) for g in gnames), dim=0)
        f_dist = self._gp.pyro_model((idx[..., None], covars), name_prefix=f"{id}_gp")

        nonfactor_plate = self._get_nonfactor_plate(nonfactor_plates)
        with pyro.plate(f"{id}_gp_batch", factor_plate.size, dim=-2):  # needs to be dim=-2 to work with GPyTorch
            f = pyro.sample(f"{id}_gp.f", f_dist)
        f = f.swapaxes(-2, -1)

        with factor_plate, nonfactor_plate:
            return dict(
                zip(
                    self._names,
                    torch.split(
                        pyro.sample(f"{id}_z", dist.Normal(f, 1 - self._gp.outputscale)),
                        tuple(gp_covariates[g].shape[0] for g in gnames),
                        dim=-2,
                    ),
                    strict=False,
                )
            )

    @pyro_method
    def guide(
        self,
        id: str,
        factor_plate: pyro.plate,
        nonfactor_plates: Mapping[str, pyro.plate],
        gp_covariates: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        gnames = list(filter(lambda x: x in gp_covariates, self._names))
        covars = torch.cat(tuple(gp_covariates[g] for g in gnames), dim=0)
        idx = torch.cat(tuple(self._idx[g].expand(gp_covariates[g].shape[0]) for g in gnames), dim=0)
        f_dist = self._gp.pyro_guide((idx[..., None], covars), name_prefix=f"{id}_gp")

        nonfactor_plate = self._get_nonfactor_plate(nonfactor_plates)
        with pyro.plate(f"{id}_gp_batch", factor_plate.size, dim=-2):  # needs to be dim=-2 to work with GPyTorch
            pyro.sample(f"{id}_gp.f", f_dist)

        with factor_plate, nonfactor_plate as index:
            return dict(
                zip(
                    self._names,
                    torch.split(
                        pyro.sample(f"{id}_z", dist.Normal(self._loc[index, :], self._scale[index, :])),
                        tuple(gp_covariates[g].shape[0] for g in gnames),
                        dim=-2,
                    ),
                    strict=False,
                )
            )

    @property
    def posterior(self) -> MeanStd:
        loc = dict(zip(self._names, torch.split(self._loc, self._sizes, dim=0), strict=False))
        scale = dict(zip(self._names, torch.split(self._scale, self._sizes, dim=0), strict=False))
        posteriors = MeanStd(loc, scale)
        for res in posteriors:
            for k, v in res.items():
                res[k] = v
        return posteriors
