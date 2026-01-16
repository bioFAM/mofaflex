import logging
import operator
from collections.abc import Mapping, Sequence
from functools import reduce
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
from ..pcgse import pcgse_test
from ..utils import MeanStd, PyroParameterDict
from .base import Prior

_logger = logging.getLogger(__name__)


class Horseshoe(Prior):
    """Horseshoe sparsity-inducing prior."""

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
        regularized: bool = True

        self._regularized = regularized

        self._global_scale_locs = PyroParameterDict()
        self._inter_scale_locs = PyroParameterDict()
        self._local_scale_locs = PyroParameterDict()
        self._caux_locs = PyroParameterDict()
        self._locs = PyroParameterDict()

        self._global_scale_scales = PyroParameterDict()
        self._inter_scale_scales = PyroParameterDict()
        self._local_scale_scales = PyroParameterDict()
        self._caux_scales = PyroParameterDict()
        self._scales = PyroParameterDict()

        ndims = abs(min(factor_dim, nonfactor_dim))
        inter_scale_shape = [1] * ndims
        inter_scale_shape[factor_dim] = n_factors

        for name in self._names:
            self._global_scale_locs[name] = PyroParam(torch.full((1,), init_loc))
            self._global_scale_scales[name] = PyroParam(
                torch.full((1,), init_scale), constraint=constraints.softplus_positive
            )
            self._inter_scale_locs[name] = PyroParam(torch.full(inter_scale_shape, init_loc))
            self._inter_scale_scales[name] = PyroParam(
                torch.full(inter_scale_shape, init_scale), constraint=constraints.softplus_positive
            )
            self._local_scale_locs[name] = PyroParam(torch.full(self._shapes[name], init_loc))
            self._local_scale_scales[name] = PyroParam(
                torch.full(self._shapes[name], init_scale), constraint=constraints.softplus_positive
            )
            self._caux_locs[name] = PyroParam(torch.full(self._shapes[name], init_loc))
            self._caux_scales[name] = PyroParam(
                torch.full(self._shapes[name], init_scale), constraint=constraints.softplus_positive
            )

            if init_tensor is not None:
                loc = init_tensor[name]["loc"]
                scale = init_tensor[name]["scale"]
            else:
                loc = torch.full(self._shapes[name], init_loc)
                scale = torch.full(self._shapes[name], init_scale)
            self._locs[name] = PyroParam(loc)
            self._scales[name] = PyroParam(scale, constraint=constraints.softplus_positive)

    def _get_prior_scale(self, name: str, **kwargs):
        return None

    def _model(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        global_scale = pyro.sample(f"global_scale_z_{name}", dist.HalfCauchy(torch.ones((1,))))
        with factor_plate:
            inter_scale = pyro.sample(f"inter_scale_z_{name}", dist.HalfCauchy(torch.ones((1,))))
            with nonfactor_plate:
                local_scale = pyro.sample(f"local_scale_z_{name}", dist.HalfCauchy(torch.ones((1,))))
                local_scale = local_scale * inter_scale * global_scale
                if self._regularized:
                    caux = pyro.sample(
                        f"caux_z_{name}", dist.InverseGamma(torch.full((1,), 0.5), torch.full((1,), 0.5))
                    )
                    c = torch.sqrt(caux)
                    if (prior_scale := self._get_prior_scale(name, **kwargs)) is not None:
                        c = c * prior_scale.reshape(self._shapes[name])
                    local_scale = (c * local_scale) / torch.sqrt(c**2 + local_scale**2)
                return pyro.sample(f"z_{name}", dist.Normal(torch.zeros((1,)), local_scale))

    def _guide(self, name: str, factor_plate: pyro.plate, nonfactor_plate: pyro.plate, **kwargs) -> torch.Tensor:
        pyro.sample(
            f"global_scale_z_{name}", dist.LogNormal(self._global_scale_locs[name], self._global_scale_scales[name])
        )
        with factor_plate:
            pyro.sample(
                f"inter_scale_z_{name}", dist.LogNormal(self._inter_scale_locs[name], self._inter_scale_scales[name])
            )
            with nonfactor_plate as index:
                local_scale_loc = self._local_scale_locs[name].index_select(nonfactor_plate.dim, index)
                local_scale_scale = self._local_scale_scales[name].index_select(nonfactor_plate.dim, index)
                pyro.sample(f"local_scale_z_{name}", dist.LogNormal(local_scale_loc, local_scale_scale))

                if self._regularized:
                    caux_loc = self._caux_locs[name].index_select(nonfactor_plate.dim, index)
                    caux_scale = self._caux_scales[name].index_select(nonfactor_plate.dim, index)
                    pyro.sample(f"caux_z_{name}", dist.LogNormal(caux_loc, caux_scale))

                return pyro.sample(
                    f"z_{name}",
                    dist.Normal(
                        self._locs[name].index_select(nonfactor_plate.dim, index),
                        self._scales[name].index_select(nonfactor_plate.dim, index),
                    ),
                )

    @property
    def posterior(self) -> MeanStd:
        posteriors = MeanStd({}, {})
        for name in self._names:
            posteriors.mean[name] = self._locs[name].squeeze(self._squeezedims)
            posteriors.std[name] = self._scales[name].squeeze(self._squeezedims)
        return posteriors


class InformedHorseshoe(Horseshoe):
    """Horseshoe prior with domain knowledge.

    Args:
        annotations_varm_key: Key in `.varm` for the feature set annotations.
        annotation_confidence: Confidence in the provided feature annotation. Must be between 0 and 1.
            Smaller values make the model more likely to add features to the annotated pathways during
            training, while larger values encourage the model to more closely adhere to the provided annotations.
    """

    _factors = False
    _weights = True
    _state_attrs = (
        "_annotation_confidence",
        "_annotations_varm_key",
        "_annotations",
        "_informed_factors_start_idx",
        "_n_informed_factors",
        "_pcgse",
    )

    def __init__(self, names: str | Sequence[str], annotations_varm_key: str, annotation_confidence: float = 0.99):
        super().__init__(names)

        self._annotations_varm_key = annotations_varm_key
        self._annotation_confidence = annotation_confidence

    def _on_train_start(
        self,
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
        init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] | None = None,
    ):
        super()._on_train_start(factor_dim, nonfactor_dim, n_factors, n_nonfactors, init_tensor)

    def _get_prior_scale(self, name: str, hs_prior_scales: dict[str, torch.Tensor], **kwargs):
        return hs_prior_scales[name]

    def get_datasets(
        self,
        data: MofaFlexDataset,
        axis: Literal[0, 1],
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
    ) -> dict[str, dict[str, np.ndarray]]:
        prior_scales = {
            name: np.clip(
                self._annotations.get(name, np.broadcast_to(0, (n_nonfactors[name], self._n_informed_factors))).astype(
                    np.float32
                )
                + (1 - self._annotation_confidence),
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

        for name in self._names:
            prior_scale = prior_scales[name]
            if factor_dim < nonfactor_dim and prior_scale.shape[0] != n_factors:
                prior_scales[name] = prior_scale.T

        return {"hs_prior_scales": prior_scales}

    def adjust_factors(self, data: MofaFlexDataset, axis: Literal[0, 1], factors: list[str]) -> list[str]:
        annotations = data.get_covariates(
            axis,
            mkey=self._annotations_varm_key,
            filter_names=self._names,
            fill_value=lambda dt: False if dt == "boolean" or dt == np.bool else pd.NA,
        )
        for name in list(annotations.keys()):
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
        self._informed_factors_start_idx = len(factors)
        self._n_informed_factors = len(self._annotations_names)
        factors.extend(self._annotations_names)

        return factors

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
    def factors_subset(self) -> slice:
        return slice(self._informed_factors_start_idx, self._informed_factors_start_idx + self._n_informed_factors)

    @Prior._api
    @property
    def n_informed_factors(self) -> int:
        """Number of informed factors."""
        return self._n_informed_factors

    @Prior._api(has_factors=False)
    def get_significant_annotations(self) -> Mapping[str, pd.DataFrame]:
        """Get the results of significance testing of annotations against factors.

        The significance testing is an implementation of PCGSE :cite:p:`pmid26300978`. While
        originally intended to assign annotations to uninformed factors, here it is used
        as a diagnostic plot to find factors that are mismatched to their annotations.

        Returns:
            PCGSE results for each view or `None` if the model does not have prior annotations.
        """
        return MappingProxyType(self._pcgse)

    @Prior._api(has_factors=True, factors_subset="factors_subset")
    @property
    def annotations(self) -> Mapping[str, pd.DataFrame]:
        """Annotation matrices for each view."""
        return MappingProxyType(self._annotations)
