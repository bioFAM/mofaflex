import logging
import operator
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Literal

import numpy as np
import pandas as pd

from ..datasets import MofaFlexDataset
from ..pcgse import pcgse_test
from ..pyro.priors import Horseshoe as PyroHorseshoe
from ..utils import MeanStd
from .base import Prior

_logger = logging.getLogger()


class InformedHorseshoe(Prior):
    _factors = False
    _weights = True
    _state_attrs = (
        "_annotations_varm_key",
        "_annotations",
        "_informed_factors_start_idx",
        "_n_informed_factors",
        "_pcgse",
    )

    def __init__(
        self,
        axis: Literal[0, 1, "samples", "features"],
        names: str | Sequence[str],
        annotations_varm_key: str | None,
        **kwargs,
    ):
        super().__init__(axis, names)
        if self.axis != 1 and annotations_varm_key is not None:
            raise ValueError("Annotations can only be applied on features.")

        self._annotations_varm_key = annotations_varm_key

    def get_datasets(self, data: MofaFlexDataset) -> None:
        annotations = data.get_covariates(
            self.axis,
            mkey=self._annotations_varm_key,
            fill_value=lambda dt: False if dt == "boolean" or dt == np.bool else pd.NA,
        )
        for name in list(annotations.keys()):
            if name not in self._names:
                _logger.warning(
                    f"Horseshoe prior required for annotations for view {name}. Annotations will be ignored."
                )
                del annotations[name]
            else:
                annot = annotations[name]
                if all(np.all((a.dtypes == np.bool) | (a.dtypes == "boolean")) for a in annot.values()):
                    annot = reduce(operator.or_, annot.values())
                else:
                    annot = (
                        pd.concat(annot, axis=0, names=["view", "feature"])
                        .groupby("feature")
                        .mean()
                        .rename_axis(index=None)
                    )
                if pd.api.types.is_integer_dtype(annot.columns.dtype):
                    self._annotations_names = [f"Informed Factor {i + 1}" for i in range(annot.shape[1])]
                else:
                    self._annotations_names = annot.columns.to_list()
                annotations[name] = annot.to_numpy()
        if len(annotations) == 0:
            raise ValueError("No annotations found.")
        self._annotations = annotations

    def adjust_factors(self, factors: list[str]) -> list[str]:
        self._informed_factors_start_idx = len(factors)
        self._n_informed_factors = len(self._annotations_names)
        factors.extend(self._annotations_names)

        return factors

    def _get_pyro_prior(self, n_factors: int, n_nonfactors: int, annotation_confidence: float = None, *args, **kwargs):
        prior_scales = {
            name: np.clip(
                self._annotations.get(name, np.broadcast_to(0, (self._n_informed_factors, n_nonfactors[name]))).astype(
                    np.float32
                )
                + (1 - annotation_confidence),
                1e-8,
                1.0,
            )
            for name in self._names
        }

        if n_factors > self._n_informed_factors:
            one = np.asarray(1, dtype=np.float32)
            prior_scales = {
                name: np.concatenate(
                    (
                        np.broadcast_to(one, (n_nonfactors[name], self._informed_factors_start_idx)),
                        scales,
                        np.broadcast_to(
                            one,
                            (
                                n_nonfactors[name],
                                n_factors - self._informed_factors_start_idx - self._n_informed_factors,
                            ),
                        ),
                    ),
                    axis=1,
                )
                for name, scales in prior_scales.items()
            }
        return PyroHorseshoe(
            self._names, *args, n_factors=n_factors, n_nonfactors=n_nonfactors, **kwargs, prior_scales=prior_scales
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
        self._pcgse = pcgse_test(
            data,
            nonnegative_weights=results_nonnegative,
            annotations={
                name: pd.DataFrame(annot, index=nonfactor_names[name], columns=self._annotations_names)
                for name, annot in self._annotations.items()
            },
            weights={
                name: pd.DataFrame(res, index=nonfactor_names[name], columns=factor_names)
                for name, res in results.mean.items()
            },
            min_size=1,
            subsample=1000,
        )

    @property
    def factors_subset(self):
        return slice(self._informed_factors_start_idx, self._informed_factors_start_idx + self._n_informed_factors)

    @Prior._api
    @property
    def n_informed_factors(self):
        """Number of informed factors."""
        return self._n_informed_factors

    @Prior._api(has_factors=False)
    def get_significant_annotations(self) -> dict[str, pd.DataFrame]:
        """Get the results of significance testing of annotations against factors.

        The significance testing is an implementation of PCGSE :cite:p:`pmid26300978`. While
        originally intended to assign annotations to uninformed factors, here it is used
        as a diagnostic plot to find factors that are mismatched to their annotations.

        Returns:
            PCGSE results for each view or `None` if the model does not have prior annotations.
        """
        return self._pcgse

    @Prior._api(has_factors=True, factors_subset="factors_subset")
    @property
    def annotations(self) -> dict[str, pd.DataFrame]:
        """Annotation matrices for each view."""
        return self._annotations
