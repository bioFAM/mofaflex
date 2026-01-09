from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ..datasets import MofaFlexDataset
from ..dist import ReinMaxBernoulli
from ..settings import settings
from ..utils import MeanStd, PyroParameterDict
from .base import Prior


class SpikeSlab(Prior):
    """Spike and slab sparsity-inducing prior."""

    _state_attrs = ("_probabilities", "_precisions")

    def _on_train_start(
        self,
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] | None = None,
    ):
        init_loc: float = 0.0
        init_scale: float = 0.1
        init_shape: float = 10.0
        init_rate: float = 10.0
        init_alpha: float = 1.0
        init_beta: float = 1.0
        init_prob: float = 0.5

        self.__shapes = PyroParameterDict()
        self._rates = PyroParameterDict()
        self._alphas = PyroParameterDict()
        self._betas = PyroParameterDict()
        self._probs = PyroParameterDict()
        self._locs = PyroParameterDict()
        self._scales = PyroParameterDict()

        ndims = abs(min(factor_dim, nonfactor_dim))
        shape = [1] * ndims
        shape[factor_dim] = n_factors

        for name in self._names:
            self.__shapes[name] = PyroParam(torch.full(shape, init_shape), constraint=constraints.softplus_positive)
            self._rates[name] = PyroParam(torch.full(shape, init_rate), constraint=constraints.softplus_positive)
            self._alphas[name] = PyroParam(torch.full(shape, init_alpha), constraint=constraints.softplus_positive)
            self._betas[name] = PyroParam(torch.full(shape, init_beta), constraint=constraints.softplus_positive)
            self._probs[name] = PyroParam(
                torch.full(self._shapes[name], init_prob), constraint=constraints.unit_interval
            )

            if init_tensor is not None:
                loc = init_tensor[name]["loc"]
                scale = init_tensor[name]["scale"]
            else:
                loc = torch.full(self._shapes[name], init_loc)
                scale = torch.full(self._shapes[name], init_scale)
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

        for name in self._names:
            precision_shape = self.__shapes[name].squeeze(self._squeezedims)
            precision_rate = self._rates[name].squeeze(self._squeezedims)
            d = dist.Gamma(concentration=precision_shape, rate=precision_rate)
            self._precisions.mean[name] = d.mean.cpu().numpy().T
            self._precisions.std[name] = d.stddev.cpu().numpy().T

            self._probabilities[name] = self._probs[name].squeeze(self._squeezedims).cpu().numpy().T

    def _model(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate:
            alpha = pyro.sample(f"alpha_z_{name}", dist.Gamma(torch.full((1,), 1e-3), torch.full((1,), 1e-3)))
            theta = pyro.sample(f"theta_z_{name}", dist.Beta(torch.ones((1,)), torch.ones((1,))))
            with nonfactor_plate:
                s = pyro.sample(f"s_z_{name}", dist.Bernoulli(theta))
                return pyro.sample(f"z_{name}", dist.Normal(torch.zeros((1,)), 1.0 / (alpha + settings.get("eps")))) * s

    def _guide(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        with factor_plate:
            pyro.sample(f"alpha_z_{name}", dist.Gamma(self.__shapes[name], self._rates[name]))
            pyro.sample(f"theta_z_{name}", dist.Beta(self._alphas[name], self._betas[name]))
            with nonfactor_plate as index:
                pyro.sample(
                    f"s_z_{name}",
                    ReinMaxBernoulli(temperature=2.0, probs=self._probs[name].index_select(nonfactor_plate.dim, index)),
                )

                return pyro.sample(
                    f"z_{name}",
                    dist.Normal(
                        self._locs[name].index_select(nonfactor_plate.dim, index),
                        self._scales[name].index_select(nonfactor_plate.dim, index),
                    ),
                )

    @property
    def learning_rate_multipliers(self) -> Iterator[tuple[str, float]]:
        yield from ((name, 10.0) for name, _ in self._probs.named_pyro_params(prefix="_probs"))

    @property
    def posterior(self) -> MeanStd:
        posteriors = MeanStd({}, {})
        for name in self._names:
            posteriors.mean[name] = self._locs[name].squeeze(self._squeezedims)
            posteriors.std[name] = self._scales[name].squeeze(self._squeezedims)
        return posteriors

    @Prior._api
    def get_sparse_a̲x̲i̲s̲_probabilities(self) -> Mapping[str, pd.DataFrame]:
        return MappingProxyType(self._probabilities)

    def _postprocess_name(
        self,
        results: MeanStd,
        moment: Literal["mean", "std"],
        name: str,
        sparse_type: Literal["raw", "mix", "thresh"] = "mix",
    ):
        cresults = getattr(results, moment)[name]
        if sparse_type == "mix":
            if moment == "mean":
                cresults = cresults * self._probabilities[name]
            else:
                p = self._probabilities[name]
                a = self._precisions.mean[name][:, None]
                cresults = np.sqrt(cresults**2 * p * (1 - p) + p * results.std[name] ** 2 + (1 - p) / a**2)
        elif sparse_type == "thresh":
            if moment == "mean":
                cresults = cresults * (cresults >= 0.5)
            else:
                cresults = 1 / self._precisions.mean[name]
        return cresults

    def postprocess_results(
        self,
        results: MeanStd,
        moment: Literal["mean", "std"],
        name: str | None = None,
        sparse_type: Literal["raw", "mix", "thresh"] = "mix",
        **kwargs,
    ) -> dict[str, NDArray[np.number]] | NDArray[np.number] | None:
        """Args.

        sparse_type: How to handle sparsity when using the spike and slab prior.

            - raw: Do nothing, return inferred values for all entries.
            - mix: Return the corresponding moment of a mixture distribution of two
              Normal distributions: One centered at 0 and the other centered at the
              inferred non-sparse value. The mixture is weighted by the inferred
              sparsity probability. This is what MOFA does.
            - thresh: Set all values with a sparsity probablity > 0.5 to 0.
        """
        if name is not None:
            if name in self._names:
                return self._postprocess_name(results, moment, name, sparse_type)
            else:
                return None
        else:
            ret = {}
            for name in self._names:
                ret[name] = self._postprocess_name(results, moment, name, sparse_type)
            return ret
