from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import BatchSampler, Dataset, RandomSampler, Sampler, StackDataset

from .utils import dataframe_to_numpy_dtypes


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


def merge_covariates(covariates: Mapping[str, Mapping[str, pd.DataFrame]]):
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
    merged_covariates = {}
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
        merged_covariates[group_name] = dataframe_to_numpy_dtypes(cov)
    return merged_covariates


class CovariatesDataset(Dataset):
    def __init__(
        self, covariates: Mapping[str, Mapping[str, pd.DataFrame | np.ndarray]], cast_to: np.number | None = np.float32
    ):
        super().__init__()
        self._covariates = covariates
        self._n_samples = max(covar.shape[0] for covar in self._covariates.values())
        self._cast_to = cast_to

    def __len__(self):
        return self._n_samples

    def __getitem__(self, idx: dict[str, int | list[int]]) -> dict[str, NDArray]:
        ret = {}
        for group_name, group_idx in idx.items():
            if group_name in self._covariates:
                arr = self._covariates[group_name]
                if isinstance(arr, pd.DataFrame):
                    arr = arr.iloc[group_idx, :]
                    if arr.dtypes.iloc[0] == "category":
                        arr = np.stack(tuple(arr[col].cat.codes.to_numpy() for col in arr.columns), axis=1)
                        if self._cast_to is not None:
                            arr = arr.astype(self._cast_to, copy=False)
                        arr[arr < 0] = np.nan
                    else:
                        arr = arr.to_numpy()
                else:
                    arr = arr[group_idx, :]
                if self._cast_to is not None:
                    arr = arr.astype(self._cast_to, copy=False)
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
