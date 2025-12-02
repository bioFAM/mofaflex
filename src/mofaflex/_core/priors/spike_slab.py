from typing import Literal

import numpy as np
import pandas as pd
import pyro.distributions as dist

from ..datasets import MofaFlexDataset
from ..utils import MeanStd
from .base import Prior


class SpikeSlab(Prior):
    _state_attrs = ("_precisions", "_probabilities")

    def on_train_end(
        self, data: MofaFlexDataset, results: MeanStd, results_nonnegative: dict[str, bool], batch_size: int
    ):
        self._precisions = MeanStd({}, {})
        self._probs = {}
        precisions = self._pyro_prior.posterior_precision
        probs = self._pyro_prior.posterior_probability

        for name in self._names:
            d = dist.Gamma(concentration=precisions.shape[name], rate=precisions.rate[name])
            mean, std, prob = d.mean.cpu().numpy(), d.stddev.cpu().numpy(), probs[name].cpu().numpy()
            if self._axis == 0:
                mean, std, prob = mean.T, std.T, prob.T
            prob = pd.DataFrame(prob, index=results.mean[name].index, columns=results.mean[name].columns)
            self._precisions.mean[name] = mean
            self._precisions.std[name] = std
            self._probs[name] = prob

    @Prior._api
    def get_sparse_a̲x̲i̲s̲_probabilities(self) -> dict[str, pd.DataFrame]:
        return self._probs

    def postprocess_results(
        self,
        results: MeanStd,
        moment: Literal["mean", "std"],
        sparse_type: Literal["raw", "mix", "thresh"] = "mix",
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        """Args.

        sparse_type: How to handle sparsity when using the spike and slab prior.

            - raw: Do nothing, return inferred values for all entries.
            - mix: Return the corresponding moment of a mixture distribution of two
              Normal distributions: One centered at 0 and the other centered at the
              inferred non-sparse value. The mixture is weighted by the inferred
              sparsity probability. This is what MOFA does.
            - thresh: Set all values with a sparsity probablity > 0.5 to 0.
        """
        ret = {}
        for name in self._names:
            cresults = getattr(results, moment)[name]
            if sparse_type == "mix":
                if moment == "mean":
                    cresults = cresults * self._probs[name]
                else:
                    p = self._probs[name]
                    a = self._precisions.mean[name][:, None]
                    cresults = np.sqrt(cresults**2 * p * (1 - p) + p * results.std[name] ** 2 + (1 - p) / a**2)
            elif sparse_type == "thresh":
                if moment == "mean":
                    cresults = cresults * (cresults >= 0.5)
                else:
                    cresults = 1 / self._precisions.mean[name]
            ret[name] = cresults
        return ret
