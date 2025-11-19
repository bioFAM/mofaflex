import logging
from collections.abc import Mapping, Sequence
from contextlib import suppress
from functools import reduce
from typing import Literal

import numpy as np
import pandas as pd

from ..datasets import MofaFlexDataset
from ..pcgse import pcgse_test
from ..pyro.priors import Horseshoe as PyroHorseshoe
from . import Prior

_logger = logging.getLogger()


class Horseshoe(Prior):
    _state_attrs = (
        "_annotations_varm_key",
        "_annotations",
        "_annotations_names",
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
        self._annotations = None
        self._annotations_names = None
        self._informed_factors_start_idx = self._n_informed_factors = None
        self._pcgse = None

    def get_datasets(self, data: MofaFlexDataset) -> None:
        if self._annotations_varm_key is not None:
            annotations, annotations_names = data.get_covariates(
                self.axis, mkey=self._annotations_varm_key, fill_value=lambda dt: False if dt == np.bool_ else np.nan
            )
            for name in list(annotations.keys()):
                if name not in self._names:
                    _logger.warning(
                        f"Horseshoe prior required for annotations for view {name}. Annotations will be ignored."
                    )
                    del annotations[name]
                    with suppress(KeyError):
                        del annotations_names[name]
                else:
                    annot = annotations[name]
                    if all(a.dtype == np.bool for a in annot.values()):
                        annot = reduce(np.logical_or, annot.values())
                    else:
                        annot = np.nanmean(np.stack(list(annot.values()), axis=1), axis=1).astype(bool)
                    annotations[name] = annot.T
            if len(annotations) > 0:
                self._annotations, self._annotations_names = annotations, annotations_names

    def adjust_factors(self, factors: list[str]) -> list[str]:
        if self._annotations is None:
            return factors
        else:
            self._informed_factors_start_idx = len(factors)
            annotated_name = next(iter(self._annotations.keys()))
            self._n_informed_factors = self._annotations[annotated_name].shape[0]
            if annotated_name in self._annotations_names:
                factors.extend(self._annotations_names[annotated_name])
            else:
                factors += [f"Informed Factor {i + 1}" for i in range(self._n_informed_factors)]

            return factors

    def pyro_prior(self, n_factors: int, n_nonfactors: int, annotation_confidence: float = None, *args, **kwargs):
        prior_scales = None
        if self._annotations is not None:
            prior_scales = {
                name: np.clip(
                    self._annotations.get(
                        name, np.broadcast_to(0, (self._n_informed_factors, n_nonfactors[name]))
                    ).astype(np.float32)
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
        results_mean: dict[str, pd.DataFrame],
        results_std: dict[str, pd.DataFrame],
        results_nonnegative: dict[str, bool],
        batch_size: int,
    ):
        if self._annotations is not None:
            self._pcgse = pcgse_test(
                data,
                nonnegative_weights=results_nonnegative,
                annotations=self.get_annotations(factor_names, nonfactor_names),
                weights=results_mean,
                min_size=1,
                subsample=1000,
            )

        self._api(self.get_annotations)
        self._api(self.get_significant_annotations)

    def get_significant_annotations(
        self, factor_names: Sequence[str], nonfactor_names: Mapping[str, Sequence[str]]
    ) -> dict[str, pd.DataFrame]:
        """Get the results of significance testing of annotations against factors.

        The significance testing is an implementation of PCGSE :cite:p:`pmid26300978`. While
        originally intended to assign annotations to uninformed factors, here it is used
        as a diagnostic plot to find factors that are mismatched to their annotations.

        Returns:
            PCGSE results for each view or `None` if the model does not have prior annotations.
        """
        if not self._pcgse:
            return None
        return self._pcgse

    def get_annotations(
        self, factor_names: Sequence[str], nonfactor_names: Mapping[str, Sequence[str]]
    ) -> dict[str, pd.DataFrame]:
        """Get the annotation matrices for each view.

        Returns:
            The annotations for each view or `None` if the model does not have prior annotations.
        """
        if not self._annotations:
            return None

        factor_names = self._subset_factor_names(factor_names)
        return {
            view_name: pd.DataFrame(annot, index=factor_names, columns=nonfactor_names[view_name])
            for view_name, annot in self._annotations.items()
        }

    def _subset_factor_names(self, factor_names):
        return factor_names[
            self._informed_factors_start_idx : self._informed_factors_start_idx + self._n_informed_factors
        ]
