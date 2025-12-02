from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import BatchSampler, Dataset, RandomSampler, Sampler, StackDataset

from .base import MofaFlexDataset


class MofaFlexBatchSampler(Sampler[Mapping[str, Sequence[int]]]):
    """A sampler for dicts.

    Given a dict with arbitrary keys and values indicating the number of data points in
    individual atasets, creates dicts of indices, such that the largest dataset is
    sampled without replacement, while for the smaller datasets multiple permutations
    are concatenated to yield the length of the largest dataset.
    """

    def __init__(
        self, n_samples: Mapping[str, int], batch_size: int, drop_last: bool = False, generator: torch.Generator = None
    ):
        super().__init__()
        self._n_samples = n_samples
        self._largest_group = max(n_samples.values())
        self._batch_size = batch_size
        self._drop_last = drop_last
        self._samplers = {
            k: BatchSampler(
                RandomSampler(range(nsamples), num_samples=self._largest_group, generator=generator),
                batch_size,
                drop_last,
            )
            for k, nsamples in self._n_samples.items()
        }

    def __len__(self):
        return (
            self._largest_group // self._batch_size
            if self._drop_last
            else (self._largest_group + self._batch_size - 1) // self._batch_size
        )

    def __iter__(self):
        iterators = {k: iter(sampler) for k, sampler in self._samplers.items()}
        for _ in range(len(self)):
            yield {k: next(sampler) for k, sampler in iterators.items()}


class CovariatesDataset(Dataset):
    def __init__(
        self,
        data: MofaFlexDataset,
        obs_key: Mapping[str, str] | None = None,
        obsm_key: Mapping[str, str] | None = None,
        group_names: str | Sequence[str] | None = None,
    ):
        super().__init__()

        if isinstance(group_names, str):
            group_names = (group_names,)
        covariates = data.get_covariates(0, obs_key, obsm_key)

        if group_names is not None:
            for group_name in list(covariates.keys()):
                if group_name not in group_names:
                    del covariates[group_name]

        # if data is categorical, get unique categories
        categories = None
        for group_covars in covariates.values():
            for view_covars in group_covars.values():
                dtypes = view_covars.dtypes
                if dtypes.nunique() > 1:
                    raise ValueError("Mixed dtypes for a covariate are not supported.")
                if dtypes.iloc[0] == "category":
                    categories = (
                        view_covars.iloc[0].cat.categories
                        if categories is None
                        else categories.union(view_covars.iloc[0].cat.categories)
                    )
        for group_covars in covariates.values():
            for view_covars in group_covars.values():
                if view_covars.dtypes.iloc[0] == "category":
                    for col in view_covars.columns:
                        view_covars[col] = view_covars[col].cat.set_categories(categories)

        # ensure the covariate value is consistent across views (nanmean or first)
        self.covariates = {}
        for group_name, group_covars in covariates.items():
            group_covariates = pd.concat(group_covars, axis=0, names=["view", "sample"])
            if (
                group_covariates.dtypes.iloc[0] == "category"
                or pd.api.types.is_integer_dtype(group_covariates.dtypes.iloc[0])
                and np.all(group_covariates.iloc[:, 0] >= 0)
            ):
                cov = group_covariates.groupby("sample").first()
            else:
                cov = group_covariates.groupby("sample").mean()
            cov.rename_axis(index=None, inplace=True)
            self.covariates[group_name] = cov

        self._n_samples = max(data.n_samples.values())
        self._cast_to = data.cast_to

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx: dict[str, int | list[int]]) -> dict[str, NDArray]:
        ret = {}
        for group_name, group_idx in idx.items():
            if group_name in self.covariates:
                group = self.covariates[group_name].iloc[group_idx, :]
                if group.dtypes.iloc[0] == "category":
                    arr = np.stack(tuple(group[col].cat.codes.to_numpy() for col in group.columns), axis=1).astype(
                        self._cast_to
                    )
                    arr[arr < 0] = np.nan
                else:
                    arr = group.to_numpy().astype(self._cast_to)
                ret[group_name] = arr
        return ret

    __getitems__ = __getitem__


class StackDataset(StackDataset):
    def __getitems__(self, idx: Sequence | Mapping):
        if isinstance(idx, Sequence):
            return super().__getitems__(idx)

        if isinstance(self.datasets, Mapping):
            return {k: self._get_items_from_dset(dataset, idx) for k, dataset in self.datasets.items()}
        else:
            return [self._get_items_from_dset(dataset, idx) for dataset in self.datasets]

    @staticmethod
    def _get_items_from_dset(dataset: Dataset, idx: dict) -> dict:
        if not callable(getattr(dataset, "__getitems__", None)):
            raise ValueError("Expected nested dataset to have a `__getitems__` method.")

        return dataset.__getitems__(idx)


class GuidingVarsDataset(StackDataset):
    def __init__(self, data: MofaFlexDataset, guiding_vars_obs_keys: Mapping[str, Mapping[str, str]] | None = None):
        datasets = {}
        for guiding_var_name, obs_key in guiding_vars_obs_keys.items():
            datasets[guiding_var_name] = CovariatesDataset(data, obs_key=obs_key)

        super().__init__(**datasets)
