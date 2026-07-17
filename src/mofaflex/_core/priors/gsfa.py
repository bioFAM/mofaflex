from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ..datasets import MofaFlexDataset, merge_covariates
from ..dist import ReinMaxBernoulli
from ..settings import settings
from ..utils import Matrix, MeanStd, PyroParameterDict
from .base import Prior


class GSFA(Prior, weights=False):
    """Guided sparse factor analysis prior for CRISPR perturbation screens.

    Args:
        targets_obsm_key: The key in `.obsm` that contains the perturbation matrix.
        s_b: The $s_b$ parameter for the hyperprior on the non-zero probability.
    """

    _state_attrs = ("_targets_obsm_key", "_s_b", "_targets", "_probabilities", "_precisions", "_beta")

    def __init__(self, names: str | Sequence[str], targets_obsm_key: str, s_b: float = 20):
        super().__init__(names)
        self._targets_obsm_key = targets_obsm_key
        self._s_b = s_b

    def get_datasets(
        self, data: MofaFlexDataset, axis: Literal[0, 1], n_factors: int, n_nonfactors: Mapping[str, int]
    ) -> dict[str, dict[str, np.ndarray]]:
        self._targets = merge_covariates(
            data.get_covariates(
                axis,
                mkey=self._targets_obsm_key,
                filter_names=self.names,
                fill_value=lambda dt: False if dt == "boolean" or dt == np.bool else pd.NA,
            )
        )
        if len(self._targets) == 0:
            raise ValueError("No perturbation targets found.")
        for target in self._targets.values():
            if pd.api.types.is_integer_dtype(target.columns):
                target.columns = "Target " + target.columns.astype(str)
        return {"perturbation_targets": self._targets}

    def on_train_start(
        self,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], Matrix[np.floating]]] | None = None,
    ):
        init_loc: float = 0.0
        init_scale: float = 0.1
        init_shape: float = 10.0
        init_rate: float = 10.0
        init_alpha: float = 1.0
        init_beta: float = 1.0
        init_prob: float = 0.2

        self._shapes = PyroParameterDict()
        self._rates = PyroParameterDict()
        self._alphas = PyroParameterDict()
        self._betas = PyroParameterDict()
        self._probs = PyroParameterDict()
        self._locs = PyroParameterDict()
        self._scales = PyroParameterDict()

        for name in self.names:
            n_perturbations = self._targets[name].shape[1]
            self._shapes[name] = PyroParam(
                torch.full((n_perturbations, 1), init_shape), constraint=constraints.softplus_positive
            )
            self._rates[name] = PyroParam(
                torch.full((n_perturbations, 1), init_rate), constraint=constraints.softplus_positive
            )
            self._alphas[name] = PyroParam(
                torch.full((n_perturbations, 1), init_alpha), constraint=constraints.softplus_positive
            )
            self._betas[name] = PyroParam(
                torch.full((n_perturbations, 1), init_beta), constraint=constraints.softplus_positive
            )
            self._probs[name] = PyroParam(
                torch.full((n_perturbations, n_factors), init_prob), constraint=constraints.unit_interval
            )

            if init_tensor is not None:
                loc = init_tensor[name]["loc"]
                scale = init_tensor[name]["scale"]

                # assume targets matrix has linearly independent cols, use pseudoinverse such that
                # Z = targets @ loc = init_tensor and Z = targets @ scale = init_tensor
                targets = torch.as_tensor(self._targets[name].to_numpy(), dtype=loc.dtype)
                ttargets = targets.T @ targets
                loc = torch.linalg.solve(ttargets, targets.T @ loc)
                scale = torch.nn.functional.softplus(torch.linalg.solve(ttargets, targets.T @ scale))
            else:
                loc = torch.full((n_perturbations, n_factors), init_loc)
                scale = torch.full((n_perturbations, n_factors), init_scale)
            self._locs[name] = PyroParam(loc)
            self._scales[name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def on_train_end(
        self,
        data: MofaFlexDataset,
        factor_names: Sequence[str],
        nonfactor_names: Mapping[str, Sequence[str]],
        results: MeanStd,
        results_nonnegative: dict[str, bool],
        batch_size: int,
    ):
        self._precisions = MeanStd({}, {})
        self._probabilities = {}
        self._beta = MeanStd({}, {})

        for name in self.names:
            precision_shape = self._shapes[name]
            precision_rate = self._rates[name]
            d = dist.Gamma(concentration=precision_shape, rate=precision_rate)
            self._precisions.mean[name] = d.mean.cpu().numpy()
            self._precisions.std[name] = d.stddev.cpu().numpy()

            self._probabilities[name] = self._probs[name].cpu().numpy()

            self._beta.mean[name] = self._locs[name].cpu().numpy()
            self._beta.std[name] = self._scales[name].cpu().numpy()

    def _get_perturbation_plate(self, id: str, name: str, perturbation_targets: Mapping[str, torch.Tensor]):
        return pyro.plate(f"{id}_plate_perturbation_targets_{name}", perturbation_targets[name].shape[1], dim=-2)

    def _model(
        self,
        id: str,
        name: str,
        factor_plate: pyro.plate,
        nonfactor_plate: pyro.plate,
        perturbation_targets: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        perturbation_plate = self._get_perturbation_plate(id, name, perturbation_targets)
        with perturbation_plate:
            p = pyro.sample(
                f"{id}_p_z_{name}", dist.Beta(torch.full((1,), self._s_b * 0.2), torch.full((1,), self._s_b * 0.8))
            )
            d = pyro.sample(f"{id}_d_z_{name}", dist.Gamma(torch.ones((1,)), torch.ones((1,))))

            with factor_plate:
                s = pyro.sample(f"{id}_s_z_{name}", dist.Bernoulli(p))
                beta = (
                    pyro.sample(
                        f"{id}_beta_z_{name}",
                        dist.Normal(torch.zeros((1,)), 1.0 / (torch.sqrt(d) + settings.get("eps"))),
                    )
                    * s
                )
        return perturbation_targets[name] @ beta

    def _guide(
        self,
        id: str,
        name: str,
        factor_plate: pyro.plate,
        nonfactor_plate: pyro.plate,
        perturbation_targets: Mapping[str, torch.Tensor],
        **kwargs,
    ) -> torch.Tensor:
        perturbation_plate = self._get_perturbation_plate(id, name, perturbation_targets)
        with perturbation_plate:
            pyro.sample(f"{id}_p_z_{name}", dist.Beta(self._alphas[name], self._betas[name]))
            pyro.sample(f"{id}_d_z_{name}", dist.Gamma(self._shapes[name], self._rates[name]))
            with factor_plate:
                pyro.sample(f"{id}_s_z_{name}", ReinMaxBernoulli(temperature=2.0, probs=self._probs[name]))
                pyro.sample(f"{id}_beta_z_{name}", dist.Normal(self._locs[name], self._scales[name]))

    @property
    def learning_rate_multipliers(self) -> Iterator[tuple[str, float]]:
        yield from ((name, 10.0) for name, _ in self._probs.named_pyro_params(prefix="_probs"))

    @property
    def posterior(self) -> MeanStd:
        posteriors = MeanStd({}, {})
        for name in self.names:
            target = torch.as_tensor(self._targets[name].to_numpy(), dtype=self._locs[name].dtype)
            posteriors.mean[name] = target @ (self._locs[name] * self._probs[name])
            posteriors.std[name] = target @ self._scales[name]
        return posteriors

    @Prior._api(has_factors=True)
    @property
    def perturbation_effects(self) -> Mapping[str, Matrix[np.floating]]:
        r"""The perturbation effect matrix :math:`\mat\beta`."""
        return MappingProxyType(self._beta.mean)

    @Prior._api(has_factors=True)
    @property
    def posterior_inclusion_probabilities(self) -> Mapping[str, Matrix[np.floating]]:
        r"""The posterior inclusion probabilities :math:`p`."""
        return MappingProxyType(self._probabilities)

    def _postprocess_name(
        self,
        results: MeanStd,
        moment: Literal["mean", "std"],
        name: str,
        sparse_type: Literal["raw", "mix", "thresh"] = "mix",
    ):
        # there is no (easy) way to simply modify the factors, so we need to recompute
        cbeta = getattr(self._beta, moment)[name]
        if sparse_type == "mix":
            if moment == "mean":
                cbeta = cbeta * self._probabilities[name]
            else:
                p = self._probabilities[name]
                a = self._precisions.mean[name][:, None]
                cbeta = np.sqrt(cbeta**2 * p * (1 - p) + p * self._beta.std[name] ** 2 + (1 - p) / a**2)
        elif sparse_type == "thresh":
            if moment == "mean":
                cbeta = cbeta * (self._probabilities[name] >= 0.5)
            else:
                mask = self._probabilities[name] < 0.5
                cbeta[mask] = 1 / self._precisions.mean[name][mask]

        return self._targets[name].to_numpy().astype(cbeta.dtype) @ cbeta

    def postprocess_results(
        self,
        results: MeanStd,
        moment: Literal["mean", "std"],
        name: str | None = None,
        sparse_type: Literal["raw", "mix", "thresh"] = "mix",
        **kwargs,
    ) -> dict[str, Matrix[np.number]] | Matrix[np.number] | None:
        """Args.

        sparse_type: How to handle sparsity when using the spike and slab prior.

            - raw: Do nothing, return inferred values for all entries.
            - mix: Return the corresponding moment of a mixture distribution of two
              Normal distributions: One centered at 0 and the other centered at the
              inferred non-sparse value. The mixture is weighted by the inferred
              sparsity probability.
            - thresh: Set all values with a sparsity probablity > 0.5 to 0.
        """
        if name is not None:
            if name in self.names:
                return self._postprocess_name(results, moment, name, sparse_type)
            else:
                return None
        else:
            ret = {}
            for name in self.names:
                ret[name] = self._postprocess_name(results, moment, name, sparse_type)
            return ret
