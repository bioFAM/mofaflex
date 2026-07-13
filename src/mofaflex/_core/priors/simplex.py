from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam, pyro_method

from ..datasets import MofaFlexDataset
from ..utils import MeanStd, PyroParameterDict, change_pyro_plate_dim
from .base import Prior


class Simplex(Prior, weights=False):
    """Dirichlet simplex prior for a prefix of the factor matrix.

    Args:
        n_simplex_factors: Number of leading factors constrained to the simplex, or "all" to
            constrain every factor after data-dependent factor extension.
        remainder_prior: Prior for the remaining factors. Can be a prior name or a public prior wrapper.
        prior_concentration: Symmetric Dirichlet prior concentration for the simplex factors. If
            None, uses 1 / n_simplex_factors (favours sparse, near-one-hot compositions).
            Larger values relax the pull towards sparse posteriors.
        init_concentration: Scale of the data-driven peak added to the per-observation posterior
            concentrations at initialization. A moderate value breaks the symmetry that otherwise
            collapses every factor to the uniform 1 / n_simplex_factors value; too large a value
            over-commits to the (arbitrary) initialization and can lock in a poor factor assignment,
            so prefer small values and let training refine them.
        concentration_lr_multiplier: Learning-rate multiplier for the per-observation concentration
            parameters, so they can escape the uniform initialization quickly enough.
    """

    _state_attrs = (
        "_n_simplex_factors_config",
        "_n_simplex_factors",
        "_n_remainder_factors",
        "_prior_concentration",
        "_init_concentration",
        "_concentration_lr_multiplier",
    )

    def __init__(
        self,
        names: str | Sequence[str],
        n_simplex_factors: int | Literal["all"],
        remainder_prior="Normal",
        prior_concentration: float | None = None,
        init_concentration: float = 5.0,
        concentration_lr_multiplier: float = 10.0,
    ):
        super().__init__(names)

        if n_simplex_factors != "all" and (not isinstance(n_simplex_factors, int) or n_simplex_factors <= 0):
            raise ValueError("'n_simplex_factors' must be a positive integer or 'all'.")

        self._n_simplex_factors_config = n_simplex_factors
        self._n_simplex_factors = 0 if n_simplex_factors == "all" else n_simplex_factors
        self._n_remainder_factors = 0
        self._remainder_prior = self._make_remainder_prior(remainder_prior)
        self._prior_concentration = prior_concentration
        self._init_concentration = init_concentration
        self._concentration_lr_multiplier = concentration_lr_multiplier

    @staticmethod
    def _slice_init_tensor(
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] | None,
        start: int,
        stop: int | None = None,
    ) -> dict[str, dict[Literal["loc", "scale"], NDArray]] | None:
        if init_tensor is None:
            return None
        return {
            name: {"loc": values["loc"][:, start:stop], "scale": values["scale"][:, start:stop]}
            for name, values in init_tensor.items()
        }

    @staticmethod
    def _plate_index(index):
        if isinstance(index, torch.Tensor):
            return index.reshape(-1)
        return index

    def _construct_factor_prior(self, spec) -> Prior:
        if isinstance(spec, str):
            prior = Prior(spec, self.names)
        elif isinstance(spec, Prior):
            prior = spec
        elif callable(spec):
            prior = spec(self.names)
        else:
            raise TypeError("Prior spec must be a prior name, prior wrapper, or Prior instance.")
        if not prior.factors_allowed():
            raise ValueError(f"The prior {prior.__class__.__name__} cannot be used for factors.")
        return prior

    def _make_remainder_prior(self, remainder_prior) -> Prior | None:
        if remainder_prior is None:
            return None
        return self._construct_factor_prior(remainder_prior)

    @property
    def _has_remainder(self) -> bool:
        return self._n_remainder_factors > 0 and self._remainder_prior is not None

    @property
    def nonnegative_factors(self) -> bool:
        """Whether all factors produced by this prior are intrinsically nonnegative."""
        if not self._has_remainder:
            return True
        return self._remainder_prior.nonnegative_factors

    def _resolve_n_simplex_factors(self, n_factors: int):
        if self._n_simplex_factors_config == "all":
            if n_factors <= 0:
                raise ValueError('n_simplex_factors="all" requires at least one factor.')
            self._n_simplex_factors = n_factors
        else:
            self._n_simplex_factors = self._n_simplex_factors_config

        if self._n_simplex_factors > n_factors:
            raise ValueError(
                "n_simplex_factors must be less than or equal to the total number of factors "
                f"({n_factors}), got {self._n_simplex_factors}."
            )

    def extend_factors(self, data: MofaFlexDataset, axis: Literal[0, 1], n_factors: int) -> Sequence[str]:
        if self._n_simplex_factors_config == "all" or self._remainder_prior is None:
            return []
        return self._remainder_prior.extend_factors(data, axis, max(n_factors - self._n_simplex_factors, 0))

    def get_datasets(
        self, data: MofaFlexDataset, axis: Literal[0, 1], n_factors: int, n_nonfactors: Mapping[str, int]
    ) -> dict[str, dict[str, np.ndarray]] | None:
        self._resolve_n_simplex_factors(n_factors)
        self._n_remainder_factors = n_factors - self._n_simplex_factors

        if self._has_remainder:
            return self._remainder_prior.get_datasets(data, axis, self._n_remainder_factors, n_nonfactors)
        return None

    def _init_concentrations(
        self, n_obs: int, init_tensor: Mapping[Literal["loc", "scale"], NDArray] | None
    ) -> torch.Tensor:
        # A uniform start makes the sampled factors indistinguishable, and together with weights that start near
        # zero it leaves the optimizer no gradient to differentiate factors. Adding a data-driven peak from the
        # factor initialization breaks that symmetry so each observation starts leaning towards a factor.
        concentrations = torch.ones(n_obs, self._n_simplex_factors)
        if init_tensor is not None and self._init_concentration > 0:
            loc = torch.as_tensor(init_tensor["loc"][:, : self._n_simplex_factors], dtype=concentrations.dtype)
            concentrations = concentrations + self._init_concentration * torch.softmax(loc, dim=-1)
        return concentrations

    def on_train_start(
        self,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] | None = None,
    ):
        self._resolve_n_simplex_factors(n_factors)
        self._n_remainder_factors = n_factors - self._n_simplex_factors

        self._concentrations = PyroParameterDict()
        for name in self.names:
            self._concentrations[name] = PyroParam(
                self._init_concentrations(n_nonfactors[name], init_tensor[name] if init_tensor is not None else None),
                constraint=constraints.softplus_positive,
            )

        if self._n_remainder_factors > 0:
            if self._remainder_prior is None:
                raise ValueError("remainder_prior cannot be None when non-simplex factors are present.")
            self._remainder_prior.on_train_start(
                self._n_remainder_factors,
                n_nonfactors,
                self._slice_init_tensor(init_tensor, self._n_simplex_factors),
            )

    def on_train_epoch_start(self, epoch: int):
        if self._has_remainder:
            self._remainder_prior.on_train_epoch_start(epoch)

    def on_train_epoch_end(self, epoch: int):
        if self._has_remainder:
            self._remainder_prior.on_train_epoch_end(epoch)

    def on_train_end(
        self,
        data: MofaFlexDataset,
        factor_names: Sequence[str],
        nonfactor_names: Mapping[str, Sequence[str]],
        results: MeanStd,
        results_nonnegative: dict[str, bool],
        batch_size: int,
    ):
        if self._has_remainder:
            self._remainder_prior.on_train_end(
                data,
                factor_names[self._n_simplex_factors :],
                nonfactor_names,
                results,
                results_nonnegative,
                batch_size,
            )

    def _remainder_factor_plate(self, id: str, factor_plate: pyro.plate) -> pyro.plate:
        return pyro.plate(f"{id}_plate_remainder_factors", self._n_remainder_factors, dim=factor_plate.dim)

    def _combine_with_remainder(
        self, simplex_factors: dict[str, torch.Tensor], remainder_factors: dict[str, torch.Tensor] | None
    ) -> dict[str, torch.Tensor]:
        if remainder_factors is None:
            return simplex_factors
        return {name: torch.cat((simplex_factors[name], remainder_factors[name]), dim=-1) for name in self.names}

    @pyro_method
    def model(
        self, id: str, factor_plate: pyro.plate, nonfactor_plates: Mapping[str, pyro.plate], **kwargs
    ) -> dict[str, torch.Tensor]:
        prior_concentration = (
            self._prior_concentration if self._prior_concentration is not None else 1.0 / self._n_simplex_factors
        )
        concentration = torch.full((self._n_simplex_factors,), prior_concentration)
        simplex_factors = {}
        with change_pyro_plate_dim(nonfactor_plates.values(), -1):
            for name in self.names:
                with nonfactor_plates[name]:
                    simplex_factors[name] = pyro.sample(f"{id}_simplex_z_{name}", dist.Dirichlet(concentration))

        remainder_factors = None
        if self._n_remainder_factors > 0:
            remainder_factors = self._remainder_prior.model(
                f"{id}_remainder", self._remainder_factor_plate(id, factor_plate), nonfactor_plates, **kwargs
            )
        return self._combine_with_remainder(simplex_factors, remainder_factors)

    @pyro_method
    def guide(
        self, id: str, factor_plate: pyro.plate, nonfactor_plates: Mapping[str, pyro.plate], **kwargs
    ) -> dict[str, torch.Tensor]:
        simplex_factors = {}
        with change_pyro_plate_dim(nonfactor_plates.values(), -1):
            for name in self.names:
                with nonfactor_plates[name] as index:
                    concentration = self._concentrations[name][self._plate_index(index), :]
                    simplex_factors[name] = pyro.sample(f"{id}_simplex_z_{name}", dist.Dirichlet(concentration))

        remainder_factors = None
        if self._n_remainder_factors > 0:
            remainder_factors = self._remainder_prior.guide(
                f"{id}_remainder", self._remainder_factor_plate(id, factor_plate), nonfactor_plates, **kwargs
            )
        return self._combine_with_remainder(simplex_factors, remainder_factors)

    @property
    def learning_rate_multipliers(self) -> Iterable[tuple[str, float]]:
        yield from (
            (name, self._concentration_lr_multiplier)
            for name, _ in self._concentrations.named_pyro_params(prefix="_concentrations")
        )
        if self._has_remainder:
            yield from (
                (f"_remainder_prior.{param_name}", multiplier)
                for param_name, multiplier in self._remainder_prior.learning_rate_multipliers
            )

    @property
    def posterior(self) -> MeanStd:
        posteriors = MeanStd({}, {})
        for name in self.names:
            simplex_dist = dist.Dirichlet(self._concentrations[name])
            posteriors.mean[name] = simplex_dist.mean
            posteriors.std[name] = simplex_dist.stddev

        if not self._has_remainder:
            return posteriors

        remainder = self._remainder_prior.posterior
        combined = MeanStd({}, {})
        for name in self.names:
            combined.mean[name] = torch.cat((posteriors.mean[name], remainder.mean[name]), dim=-1)
            simplex_std = posteriors.std.get(name)
            remainder_std = remainder.std.get(name)
            if simplex_std is not None or remainder_std is not None:
                if simplex_std is None:
                    simplex_std = torch.zeros_like(posteriors.mean[name])
                if remainder_std is None:
                    remainder_std = torch.zeros_like(remainder.mean[name])
                combined.std[name] = torch.cat((simplex_std, remainder_std), dim=-1)
        return combined

    def _save(self) -> dict:
        return {
            "simplex_block": {},
            "remainder_prior": self._remainder_prior.save() if self._has_remainder else None,
        }

    def _load(
        self,
        state: Mapping,
        *,
        map_location=None,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        **kwargs,
    ):
        if not hasattr(self, "_n_simplex_factors_config"):
            self._n_simplex_factors_config = self._n_simplex_factors
        self._resolve_n_simplex_factors(n_factors)
        self._n_remainder_factors = n_factors - self._n_simplex_factors
        self._remainder_prior = (
            Prior.load(
                state["remainder_prior"],
                map_location=map_location,
                n_factors=self._n_remainder_factors,
                n_nonfactors=n_nonfactors,
            )
            if state.get("remainder_prior") is not None
            else None
        )
