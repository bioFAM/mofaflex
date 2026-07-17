from collections.abc import Iterator, Mapping, Sequence
from types import MappingProxyType
from typing import Literal

import numpy as np
import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.nn import PyroParam

from ..datasets import MofaFlexDataset
from ..dist import ReinMaxBernoulli
from ..settings import settings
from ..utils import Matrix, MeanStd, PyroParameterDict
from .base import Prior


class SpikeSlab(Prior):
    r"""Spike and slab sparsity-inducing prior.

    Args:
        background_is_gaussian: Whether the background distribution should be a Gaussian centered at 0.
            In that case the prior is a mixture of Normal distributions.
        init_prob: The initialization value for the foreground probability :math:`\theta`. Set to `0.2` to
            be consistent with GSFA :cite:p:`pmid37770710`.
        psi_prior_param: The value for shape and rate of the Gamma distribution for :math:`\psi`. Set to `1.`
            to be consistent with GSFA :cite:p:`pmid37770710`.
        theta_prior_param_alpha: The value for the :math:`\alpha` parameter of the Beta distribution used to
            sample the foreground probabilities :math:`\theta`. Set to `10.` to be  consistent with GSFA
            :cite:p:`pmid37770710`.
        theta_prior_param_beta: The value for the :math:`\beta` parameter of the Beta distribution used to
            sample the foreground probabilities :math:`\theta`. Set to `40.` to be  consistent with GSFA
            :cite:p:`pmid37770710`.
    """

    _state_attrs = (
        "_background_is_gaussian",
        "_init_prob",
        "_alpha_prior_param",
        "_theta_prior_param_alpha",
        "_theta_prior_param_beta",
        "_probabilities",
        "_precisions",
        "_scales_background",
    )

    def __init__(
        self,
        names: str | Sequence[str],
        background_is_gaussian: bool = False,
        init_prob: float = 0.5,
        psi_prior_param: float = 1e-3,
        theta_prior_param_alpha: float = 1.0,
        theta_prior_param_beta: float = 1.0,
    ):
        super().__init__(names)
        self._background_is_gaussian = background_is_gaussian
        self._init_prob = init_prob
        self._alpha_prior_param = psi_prior_param
        self._theta_prior_param_alpha = theta_prior_param_alpha
        self._theta_prior_param_beta = theta_prior_param_beta

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
        init_background_scale: float = 0.5

        self._shapes = PyroParameterDict()
        self._rates = PyroParameterDict()
        self._alphas = PyroParameterDict()
        self._betas = PyroParameterDict()
        self._probs = PyroParameterDict()
        self._locs = PyroParameterDict()
        self._scales = PyroParameterDict()

        if self._background_is_gaussian:
            self._c_shapes = PyroParameterDict()
            self._c_rates = PyroParameterDict()
            self._scales_background = PyroParameterDict()

        for name in self.names:
            self._shapes[name] = PyroParam(
                torch.full((n_factors,), init_shape), constraint=constraints.softplus_positive
            )
            self._rates[name] = PyroParam(torch.full((n_factors,), init_rate), constraint=constraints.softplus_positive)
            self._alphas[name] = PyroParam(
                torch.full((n_factors,), init_alpha), constraint=constraints.softplus_positive
            )
            self._betas[name] = PyroParam(torch.full((n_factors,), init_beta), constraint=constraints.softplus_positive)
            self._probs[name] = PyroParam(
                torch.full((n_nonfactors[name], n_factors), self._init_prob), constraint=constraints.unit_interval
            )

            if init_tensor is not None:
                loc = init_tensor[name]["loc"]
                scale = init_tensor[name]["scale"]
            else:
                loc = torch.full((n_nonfactors[name], n_factors), init_loc)
                scale = torch.full((n_nonfactors[name], n_factors), init_scale)
            self._locs[name] = PyroParam(loc)
            self._scales[name] = PyroParam(scale, constraint=constraints.softplus_positive)

            if self._background_is_gaussian:
                self._c_shapes[name] = PyroParam(
                    torch.full((n_factors,), init_shape), constraint=constraints.softplus_positive
                )
                self._c_rates[name] = PyroParam(
                    torch.full((n_factors,), init_rate), constraint=constraints.softplus_positive
                )
                self._scales_background[name] = PyroParam(
                    torch.full((n_nonfactors[name], n_factors), init_background_scale),
                    constraint=constraints.softplus_positive,
                )

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
        scales_background = {}
        self._probabilities = {}

        for name in self.names:
            precision_shape = self._shapes[name]
            precision_rate = self._rates[name]
            d = dist.Gamma(concentration=precision_shape, rate=precision_rate)
            self._precisions.mean[name] = d.mean.cpu().numpy()
            self._precisions.std[name] = d.stddev.cpu().numpy()
            if self._background_is_gaussian:
                scales_background[name] = self._scales_background[name].cpu().numpy()

            self._probabilities[name] = self._probs[name].cpu().numpy()
        if self._background_is_gaussian:
            del self._scales_background
            self._scales_background = scales_background

    def _model(
        self, id: str, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs
    ) -> torch.Tensor:
        with factor_plate:
            alpha = pyro.sample(
                f"{id}_alpha_z_{name}",
                dist.Gamma(torch.full((1,), self._alpha_prior_param), torch.full((1,), self._alpha_prior_param)),
            )
            theta = pyro.sample(
                f"{id}_theta_z_{name}",
                dist.Beta(
                    torch.full((1,), self._theta_prior_param_alpha), torch.full((1,), self._theta_prior_param_beta)
                ),
            )
            if self._background_is_gaussian:
                c = pyro.sample(f"{id}_c_z_{name}", dist.Gamma(torch.full((1,), 3.0), torch.full((1,), 0.5)))

            with nonfactor_plate:
                s = pyro.sample(f"{id}_s_z_{name}", dist.Bernoulli(theta))
                res = (
                    pyro.sample(
                        f"{id}_z_{name}",
                        dist.Normal(torch.zeros((1,)), 1.0 / (torch.sqrt(alpha) + settings.get("eps"))),
                    )
                    * s
                )
                if self._background_is_gaussian:
                    res += pyro.sample(
                        f"{id}_background_z_{name}",
                        dist.Normal(torch.zeros((1,)), 1.0 / (torch.sqrt(alpha * c) + settings.get("eps"))),
                    ) * (1 - s)
        return res

    def _guide(
        self, id: str, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs
    ) -> torch.Tensor:
        with factor_plate:
            pyro.sample(f"{id}_alpha_z_{name}", dist.Gamma(self._shapes[name], self._rates[name]))
            pyro.sample(f"{id}_theta_z_{name}", dist.Beta(self._alphas[name], self._betas[name]))
            if self._background_is_gaussian:
                pyro.sample(f"{id}_c_z_{name}", dist.Gamma(self._c_shapes[name], self._c_rates[name]))

            with nonfactor_plate as index:
                pyro.sample(f"{id}_s_z_{name}", ReinMaxBernoulli(temperature=2.0, probs=self._probs[name][index, :]))
                pyro.sample(f"{id}_z_{name}", dist.Normal(self._locs[name][index, :], self._scales[name][index, :]))
                if self._background_is_gaussian:
                    pyro.sample(
                        f"{id}_background_z_{name}",
                        dist.Normal(torch.zeros((1,)), self._scales_background[name][index, :]),
                    )

    @property
    def learning_rate_multipliers(self) -> Iterator[tuple[str, float]]:
        yield from ((name, 10.0) for name, _ in self._probs.named_pyro_params(prefix="_probs"))

    @property
    def posterior(self) -> MeanStd:
        posteriors = MeanStd({}, {})
        for name in self.names:
            posteriors.mean[name] = self._locs[name]
            posteriors.std[name] = self._scales[name]
        return posteriors

    @Prior._api(has_factors=True)
    @property
    def sparse_a̲x̲i̲s̲_probabilities(self) -> Mapping[str, Matrix[np.floating]]:
        r"""The posterior probabilities :math:`\theta` that the value is sampled from the foreground distribution."""
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
                bgscale = (
                    self._scales_background[name] ** 2
                    if self._background_is_gaussian
                    else 1 / self._precisions.mean[name] ** 2
                )
                cresults = np.sqrt(cresults**2 * p * (1 - p) + p * results.std[name] ** 2 + (1 - p) * bgscale)
        elif sparse_type == "thresh":
            if moment == "mean":
                cresults = cresults * (self._probabilities[name] >= 0.5)
            else:
                bgscale = (
                    self._scales_background[name] if self._background_is_gaussian else 1 / self._precisions.mean[name]
                )
                mask = self._probabilities[name] < 0.5
                cresults[mask] = bgscale[mask]
        return cresults

    def postprocess_results(
        self,
        results: MeanStd,
        moment: Literal["mean", "std"],
        name: str | None = None,
        sparse_type: Literal["raw", "mix", "thresh"] = "mix",
        **kwargs,
    ) -> dict[str, Matrix[np.number]] | Matrix[np.number] | None:
        """Args:
        sparse_type: How to handle sparsity when using the spike and slab prior.

            - raw: Do nothing, return inferred values for all entries.
            - mix: Return the corresponding moment of a mixture distribution of two
              Normal distributions: One centered at 0 and the other centered at the
              inferred non-sparse value. The mixture is weighted by the inferred
              sparsity probability. This is what MOFA does.
            - thresh: Set all values with a sparsity probablity > 0.5 to 0.
        """  # noqa: D205
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
