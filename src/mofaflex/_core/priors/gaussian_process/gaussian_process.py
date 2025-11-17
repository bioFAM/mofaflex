from collections.abc import Sequence
from typing import Literal

from ...datasets import CovariatesDataset, MofaFlexDataset
from ...pyro.priors import GP as PyroGP
from .. import Prior


class GaussianProcess(Prior):
    def __init__(
        self,
        axis: Literal[0, 1, "samples", "features"],
        names: str | Sequence[str] | None,
        covariates_obs_key: str | Sequence[str] | None = None,
        covariates_obsm_key: str | Sequence[str] | None = None,
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

    def get_datasets(self, data: MofaFlexDataset) -> dict[str, CovariatesDataset]:
        dset = CovariatesDataset(data, self._obs_key, self._obm_key, self._names)
        self._covariates = dset.covariates
        self._covariates_names = dset.covariates_names
        return {"gp_covariates": dset}

    def pyro_prior(self, *args, **kwargs):
        return PyroGP(self._names, *args, **kwargs)
