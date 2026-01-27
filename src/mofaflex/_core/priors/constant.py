from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
import pandas as pd
import pyro
import torch

from ..datasets import MofaFlexDataset
from ..utils import MeanStd
from .base import Prior


class Constant(Prior):
    """Pseudo-prior that always returns the same constant values.

    When used for weights, this can be used to project new data into an already existing latent space.

    Args:
        const_values: The constant values to use for each group/view. Must have factors in columns and
            samples/features in rows. The number of factors must match the configuration of the model
            this prior is being used with. Sample/feature names must be a superset of the sample/feature
            names of the data this prior is being used with.
    """

    def __init__(self, names: str | Sequence[str], const_values: Mapping[str, pd.DataFrame]):
        super().__init__(names)
        if len(difference := set(names) - const_values.keys()) > 0:
            raise ValueError(f"The mapping given for 'const_values' is missing entries for {', '.join(difference)}.")
        if len({const.shape[1] for const in const_values.values()}) > 1:
            raise ValueError("The provided 'const_values' have different numbers of factors")
        self._const_values = const_values

    def get_datasets(
        self,
        data: MofaFlexDataset,
        axis: Literal[0, 1],
        factor_dim: int,
        nonfactor_dim: int,
        n_factors: int,
        n_nonfactors: Mapping[str, int],
    ) -> dict[str, dict[str, pd.DataFrame | np.ndarray]]:
        data_names = data.get_names(axis)
        vals = {}
        tvals = {}
        for name in self.names:
            const_val = self._const_values[name]
            if (const_n_factors := const_val.shape[1]) != n_factors:
                raise ValueError(
                    f"The constant values for '{name}' have the wrong number of factors: Expected {n_factors}, got {const_n_factors}."
                )

            val = const_val.loc[data_names[name], :].to_numpy()
            vals[name] = val
            tvals[name] = torch.as_tensor(
                val.T if factor_dim < nonfactor_dim else val, device="cpu"
            )  # for post-training processing in the MofaFlex term
        self._const_values = tvals
        return {"const": vals}

    def _model(
        self,
        id: str,
        name: str,
        factor_plate: pyro.plate,
        nonfactor_plate: pyro.plate,
        const: Mapping[str, torch.Tensor],
        **kwargs,
    ):
        return self._reshape_tensor_to_batch(const[name], name, factor_plate, nonfactor_plate)

    _guide = _model

    @property
    def posterior(self) -> MeanStd:
        return MeanStd(self._const_values, {})
