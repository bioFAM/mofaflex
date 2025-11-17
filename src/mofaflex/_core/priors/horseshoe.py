import logging
from collections.abc import Sequence
from functools import reduce
from typing import Literal

import numpy as np

from ..datasets import MofaFlexDataset
from ..pyro.priors import Horseshoe as PyroHorseshoe
from . import Prior

_logger = logging.getLogger()


class Horseshoe(Prior):
    def __init__(
        self,
        axis: Literal[0, 1, "samples", "features"],
        names: str | Sequence[str] | None,
        annotations_varm_key: str | None = None,
    ):
        super().__init__(axis, names)
        if self._axis != 1 and annotations_varm_key is not None:
            raise ValueError("Annotations can only be applied on features.")

        self._annotations_varm_key = annotations_varm_key
        self._annotations = None
        self._annotations_names = None
        self._informed_factors = None

    def get_datasets(self, data: MofaFlexDataset) -> None:
        if self._annotations_varm_key is not None:
            annotations, annotations_names = data.get_covariates(
                self._axis, mkey=self._annotations_varm_key, fill_value=lambda dt: False if dt == np.bool_ else np.nan
            )
            for name in list(annotations.keys()):
                if name not in self._names:
                    _logger.warning(
                        f"Horseshoe prior required for annotations for view {name}. Annotations will be ignored."
                    )
                    del annotations[name]
                    try:
                        del annotations_names[name]
                    except KeyError:
                        pass
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
            n_factors = factors.size
            annotated_name = next(iter(self._annotations.keys()))
            n_informed_factors = self._annotations[annotated_name].shape[0]
            if annotated_name in self._annotations_names:
                factors.extend(self._annotations_names[annotated_name])
            else:
                factors += [f"Informed Factor {i + 1}" for i in range(n_informed_factors)]

            self._informed_factors = (n_factors, n_informed_factors)
            return factors

    def pyro_prior(self, n_factors, n_nonfactors, annotation_confidence=None, *args, **kwargs):
        prior_scales = None
        if self._annotations is not None:
            prior_scales = {
                name: np.clip(
                    self._annotations.get(
                        name, np.broadcast_to(0, (self._informed_factors[1], n_nonfactors[name]))
                    ).astype(np.float32)
                    + (1 - annotation_confidence),
                    1e-8,
                    1.0,
                )
                for name in self._names
            }

            if n_factors > self._informed_factors[1]:
                one = np.asarray(1, dtype=np.float32)
                prior_scales = {
                    name: np.concatenate(
                        (
                            np.broadcast_to(one, (self._informed_factors[0], n_nonfactors[name])),
                            scales,
                            np.broadcast_to(one, (n_nonfactors - sum(self._informed_factors), n_nonfactors[name])),
                        ),
                        axis=0,
                    )
                    for name, scales in prior_scales.items()
                }
        return PyroHorseshoe(
            self._names, *args, n_factors=n_factors, n_nonfactors=n_nonfactors, **kwargs, prior_scales=prior_scales
        )
