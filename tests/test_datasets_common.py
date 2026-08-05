import numpy as np
import pytest

from mofaflex._core.datasets import MofaFlexDataset, merge_covariates


@pytest.mark.parametrize("axis", (0, 1))
def test_merge_covariates_preserves_axis_order(axis, random_adata):
    """merge_covariates must keep rows in the dataset's sample/feature order, not lexically sorted."""

    adata = random_adata("Normal", 500, 500)
    dataset = MofaFlexDataset(adata)

    covars = dataset.get_covariates(axis, mkey="covar_array")
    merged = next(iter(merge_covariates(covars).values()))
    canonical = next(iter(dataset.get_names(axis).values()))

    assert np.all(merged.index == canonical)
    assert np.all(merged.to_numpy() == next(iter(next(iter(covars.values())).values())))
