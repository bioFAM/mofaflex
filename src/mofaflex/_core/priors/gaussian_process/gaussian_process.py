import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import torch
from dtw import dtw

from ...datasets import CovariatesDataset, MofaFlexDataset
from ...pyro.priors import GP as PyroGP
from ...utils import MeanStd, Options, pickle_torch_state, unpickle_torch_state
from .. import Prior
from .gp import GP

_logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class SmoothOptions(Options):
    """Options for Gaussian processes."""

    n_inducing: int = 100
    """Number of inducing points."""

    kernel: Literal["RBF", "Matern"] = "RBF"
    """Kernel function to use."""

    mefisto_kernel: bool = True
    """Whether to use the MEFISTO group covariance kernel or treat groups independently."""

    independent_lengthscales: bool = False
    """Whether to use a separate lengthscale per covariate dimension."""

    group_covar_rank: int = 1
    """Rank of the group correlation matrix. Only relevant if `mefisto_kernel=True`."""

    warp_groups: Sequence[str] = field(default_factory=list)
    """List of groups to apply dynamic time warping to."""

    warp_interval: int = 20
    """Apply dynamic time warping every `warp_interval` epochs."""

    warp_open_begin: bool = True
    """Perform open-ended alignment."""

    warp_open_end: bool = True
    """Perform open-ended alignment."""

    warp_reference_group: str | None = None
    """Reference group to align the others to. Defaults to the first group of `warp_groups`."""

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.warp_groups, str):
            self.warp_groups = [self.warp_groups]
        else:
            self.warp_groups = list(self.warp_groups)  # in case the user passed a tuple here, we need a list for saving


class GaussianProcess(Prior):
    _state_attrs = "_obs_key", "_obsm_key", "_covariates", "_covariates_names", "_orig_covariates", "_warp_groups_order"

    def __init__(
        self,
        axis: Literal[0, 1, "samples", "features"],
        names: str | Sequence[str] | None,
        covariates_obs_key: str | Sequence[str] | None = None,
        covariates_obsm_key: str | Sequence[str] | None = None,
        options: SmoothOptions = None,
        **kwargs,
    ):
        super().__init__(axis, names)

        if covariates_obs_key is None and covariates_obsm_key is None:
            raise ValueError("Neither `covariates_obs_key` nor covariates_obsm_key` given.")
        if covariates_obs_key is not None and covariates_obsm_key is not None:
            raise ValueError("Provide either `covariates_obs_key` or `covariates_obsm_key`, but not both.")

        self._obs_key = covariates_obs_key
        self._obsm_key = covariates_obsm_key
        self._covariates = None
        self._covariates_names = None
        self._orig_covariates = None
        self._opts = options if options is not None else SmoothOptions()
        self._warp_groups_order = None

        self._gp = None
        self._gps = None

        self._pyro_prior = None

    def get_datasets(self, data: MofaFlexDataset) -> dict[str, CovariatesDataset]:
        dset = CovariatesDataset(data, self._obs_key, self._obsm_key, self._names)
        self._covariates = dset.covariates
        self._covariates_names = dset.covariates_names
        return {"gp_covariates": dset}

    def pyro_prior(self, n_factors: int, *args, **kwargs):
        if len(self._opts.warp_groups) > 1:
            if not set(self._opts.warp_groups) <= set(self._names):
                raise ValueError(
                    "The set of groups with dynamic time warping must be a subset of groups with a GP factor prior."
                )
            self._warp_groups_order = {}
            for g in self._opts.warp_groups:
                ccov = self._covariates[g].squeeze()
                if ccov.ndim > 1:
                    raise ValueError(
                        f"Warping can only be performed with 1D covariates, but the covariate for group {g} has {ccov.ndim} dimensions."
                    )
                self._warp_groups_order[g] = ccov.argsort()
            self._orig_covariates = {g: c.copy() for g, c in self._covariates.items()}

            if self._opts.warp_reference_group is None:
                self._opts.warp_reference_group = self._opts.warp_groups[0]
        elif len(self._opts.warp_groups) == 1:
            _logger.warning("Need at least 2 groups for warping, but only one was given. Ignoring warping.")
            self._opts.warp_groups = []

        self._init_gp(n_factors)

        self._pyro_prior = PyroGP(self._names, *args, n_factors=n_factors, gp=self._gp, **kwargs)
        return self._pyro_prior

    def _init_gp(self, n_factors: int):
        self._gp = GP(
            n_inducing=self._opts.n_inducing,
            covariates=self._covariates.values(),
            n_factors=n_factors,
            n_groups=len(self._names),
            kernel=self._opts.kernel,
            independent_lengthscales=self._opts.independent_lengthscales,
            rank=self._opts.group_covar_rank,
            use_mefisto_kernel=self._opts.mefisto_kernel,
        )

    @torch.inference_mode()
    def on_train_epoch_end(self, epoch: int):
        if len(self._opts.warp_groups) and epoch > 0 and not epoch % self._opts.warp_interval:
            factormeans = {
                group_name: mean.cpu().numpy() for group_name, mean in self._pyro_prior.posterior.mean.items()
            }  # TODO: investigate how warping interacts with non-negativity
            refgroup = self._opts.warp_reference_group
            reffactormeans = factormeans[refgroup].mean(axis=0)
            refidx = self._warp_groups_order[refgroup]
            for g in self._opts.warp_groups[1:]:
                idx = self._warp_groups_order[g]
                alignment = dtw(
                    reffactormeans[refidx],
                    factormeans[g][:, idx].mean(axis=0),
                    open_begin=self._opts.warp_open_begin,
                    open_end=self._opts.warp_open_end,
                    step_pattern="asymmetric",
                )
                self._covariates[g] = self._orig_covariates[g].copy()
                self._covariates[g][idx[alignment.index2], 0] = self._orig_covariates[refgroup][
                    refidx[alignment.index1], 0
                ]
            self._gp.update_inducing_points(self._covariates.values())

    def on_train_end(
        self,
        data: MofaFlexDataset,
        factor_names: Sequence[str],
        nonfactor_names: Mapping[str, Sequence[str]],
        results_mean: dict[str, pd.DataFrame],
        results_std: dict[str, pd.DataFrame],
        results_nonnegative: dict[str, bool],
        batch_size: int,
    ):
        self._gps = self._get_gps(self._covariates, batch_size)

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
    def get_gps(  # noqa D417
        self,
        factor_names: Sequence[str],
        nonfactor_names: Mapping[str, Sequence[str]],
        moment: Literal["mean", "std"] = "mean",
        x: Mapping[str, np.ndarray | torch.Tensor] | None = None,
        batch_size: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Get all latent functions.

        Args:
             moment: Which moment of the posterior distribution to return.
             x: Covariate values for each group. If `None`, will return latent function values at
                 covariate coordinates used for training.
             batch_size: Minibatch size. Only has an effect if `x` is not `None`. Defaults to the
                 minibatch size used for training.
        """
        gps = getattr(self._gps if x is None else self._get_gps(x, batch_size), moment)
        return {
            group_name: pd.DataFrame(
                group_f, index=nonfactor_names[group_name] if x is None else None, columns=factor_names
            )
            for group_name, group_f in gps.items()
        }

    def _save(self) -> dict:
        state = {}
        state["opts"] = asdict(self._opts)
        state["gps"] = self._gps._asdict()
        if self._gp is not None:
            state["gp_state"] = pickle_torch_state(self._gp.state_dict())
        return state

    def _load(self, state: dict, n_factors: int, n_nonfactors: Mapping[str, int], map_location=None):
        self._opts = SmoothOptions(**state["opts"])
        self._gps = MeanStd(**state["gps"])
        self._init_gp(n_factors)
        try:
            self._gp.load_state_dict(unpickle_torch_state(state["gp_state"], map_location=map_location))
        except KeyError:
            pass
