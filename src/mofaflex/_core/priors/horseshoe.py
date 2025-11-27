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


class Horseshoe(Prior):
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
        names: str | Sequence[str] | None,
        annotations_varm_key: str | None = None,
        **kwargs,
    ):
        super().__init__(axis, names)
        if self.axis != 1 and annotations_varm_key is not None:
            raise ValueError("Annotations can only be applied on features.")

        self._annotations_varm_key = annotations_varm_key

    def get_datasets(self, data: MofaFlexDataset) -> None:
        if self._annotations_varm_key is not None:
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
                    annot = annot.T
                    if pd.api.types.is_integer_dtype(annot.index.dtype):
                        annot.index = [f"Informed Factor {i + 1}" for i in range(annot.shape[0])]
                    annotations[name] = annot
            if len(annotations) > 0:
                self._annotations = annotations

    def adjust_factors(self, factors: list[str]) -> list[str]:
        if self._annotations is None:
            return factors
        else:
            self._informed_factors_start_idx = len(factors)
            annotated_name = next(iter(self._annotations.keys()))
            self._n_informed_factors = self._annotations[annotated_name].shape[0]
            factors.extend(self._annotations[annotated_name].index.to_list())

            return factors

    def _get_pyro_prior(self, n_factors: int, n_nonfactors: int, annotation_confidence: float = None, *args, **kwargs):
        prior_scales = None
        if self._annotations is not None:
            annotations = {name: annot.to_numpy() for name, annot in self._annotations.items()}
            prior_scales = {
                name: np.clip(
                    annotations.get(name, np.broadcast_to(0, (self._n_informed_factors, n_nonfactors[name]))).astype(
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
                            np.broadcast_to(one, (self._informed_factors_start_idx, n_nonfactors[name])),
                            scales,
                            np.broadcast_to(
                                one,
                                (
                                    n_factors - self._informed_factors_start_idx - self._n_informed_factors,
                                    n_nonfactors[name],
                                ),
                            ),
                        ),
                        axis=0,
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
        if self._annotations is not None:
            self._pcgse = pcgse_test(
                data,
                nonnegative_weights=results_nonnegative,
                annotations=self._annotations,
                weights={
                    view_name: pd.DataFrame(res, index=factor_names, columns=nonfactor_names[view_name])
                    for view_name, res in results.mean.items()
                },
                min_size=1,
                subsample=1000,
            )

        self._api("annotations", has_factors=True)
        self._api(self.get_significant_annotations)
        self._api("n_informed_factors")

    @property
    def n_informed_factors(self):
        """Number of informed factors."""
        return self._n_informed_factors

    def get_significant_annotations(self) -> dict[str, pd.DataFrame]:
        """Get the results of significance testing of annotations against factors.

        The significance testing is an implementation of PCGSE :cite:p:`pmid26300978`. While
        originally intended to assign annotations to uninformed factors, here it is used
        as a diagnostic plot to find factors that are mismatched to their annotations.

        Returns:
            PCGSE results for each view or `None` if the model does not have prior annotations.
        """
        return self._pcgse

    @property
    def annotations(self) -> dict[str, pd.DataFrame]:
        """Annotation matrices for each view."""
        return self._annotations

    def _subset_factor_names(self, factor_names):
        return factor_names[
            self._informed_factors_start_idx : self._informed_factors_start_idx + self._n_informed_factors
        ]
