from functools import reduce

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from mofaflex import settings
from mofaflex._core import MofaFlexDataset
from mofaflex._core.utils import filter_constant_features, sample_all_data_as_one_batch


@pytest.fixture(scope="module", params=[np.asarray, csc_array, csc_matrix, csr_array, csr_matrix])
def array(rng, request):
    def make_array(ncols, constant_cols, fill_value=None):
        arr = rng.poisson(0.5, size=(100, ncols))
        for col in constant_cols:
            arr[:, col] = arr[0, col] if fill_value is None else fill_value
        return request.param(arr)

    return make_array


@pytest.fixture(scope="module", params=[0, 1, 2, 3, "all"])
def array1_n_constant_cols(request):
    return request.param


@pytest.fixture(scope="module", params=[0, 1, 2, 3, "all"])
def array2_n_constant_cols(request):
    return request.param


@pytest.fixture(scope="module", params=["groups", "views"])
def arrays_are(request):
    return request.param


@pytest.fixture(scope="module", params=["partial", "full"])
def adata_dict(rng, array, array1_n_constant_cols, array2_n_constant_cols, arrays_are, create_adata, request):
    arrays_ncols = rng.integers(30, 60, size=2)
    genes = np.asarray([f"gene_{i}" for i in range(arrays_ncols.max())])

    array1_genes = pd.Index(rng.choice(genes, size=arrays_ncols[0], replace=False))
    array2_genes = pd.Index(rng.choice(genes, size=arrays_ncols[1], replace=False))

    genes_intersection = array1_genes.intersection(array2_genes)

    if array1_n_constant_cols == "all":
        array1_constant_cols = np.arange(genes_intersection.size)
    elif array1_n_constant_cols > 0:
        array1_constant_cols = rng.choice(arrays_ncols.min(), size=array1_n_constant_cols, replace=False)
    else:
        array1_constant_cols = []

    if array2_n_constant_cols == "all":
        array2_constant_cols = np.arange(genes_intersection.size)
    elif array2_n_constant_cols > 0:
        if request.param == "partial":
            if len(array1_constant_cols) > 0:
                choices = np.concatenate(
                    (
                        np.arange(array1_constant_cols[0]),
                        np.arange(array1_constant_cols[0] + 1, genes_intersection.size),
                    )
                )
                array2_constant_cols = np.concatenate(
                    (array1_constant_cols[:1], rng.choice(choices, size=array2_n_constant_cols - 1, replace=False))
                )
            else:
                array2_constant_cols = rng.choice(genes_intersection.size, size=array2_n_constant_cols, replace=False)
        else:
            if len(array1_constant_cols) > 0:
                if array2_n_constant_cols < len(array1_constant_cols):
                    array2_constant_cols = array1_constant_cols
                else:
                    choices = np.setdiff1d(np.arange(genes_intersection.size), array1_constant_cols)
                    array2_constant_cols = np.concatenate(
                        (
                            array1_constant_cols,
                            rng.choice(choices, size=array2_n_constant_cols - len(array1_constant_cols), replace=False),
                        )
                    )
            else:
                array2_constant_cols = rng.choice(genes_intersection.size, size=array2_n_constant_cols, replace=False)
    else:
        array2_constant_cols = []

    array1_constant_genes = genes_intersection[array1_constant_cols]
    array2_constant_genes = genes_intersection[array2_constant_cols]

    array1_constant_cols = [array1_genes.get_loc(gene) for gene in array1_constant_genes]
    array2_constant_cols = [array2_genes.get_loc(gene) for gene in array2_constant_genes]

    array1 = array(arrays_ncols[0], array1_constant_cols, fill_value=42)
    array2 = array(arrays_ncols[1], array2_constant_cols, fill_value=42)

    adata1 = create_adata(
        array1, var_names=array1_genes, obs_names=[f"group_1_obs_{i}" for i in range(array1.shape[0])]
    )
    adata2 = create_adata(
        array2, var_names=array2_genes, obs_names=[f"group_2_obs_{i}" for i in range(array2.shape[0])]
    )

    if arrays_are == "groups":
        adata_dict = {"group1": {"view1": adata1}, "group2": {"view1": adata2}}
        constgenes = {"group1": array1_constant_genes, "group2": array2_constant_genes}
    else:
        adata_dict = {"group1": {"view1": adata1, "view2": adata2}}
        constgenes = {"view1": array1_constant_genes, "view2": array2_constant_genes}
    return adata_dict, constgenes


@pytest.mark.parametrize("usedask", [False, True])
def test_remove_constant_features(adata_dict, arrays_are, usedask):
    adata_dict, constgenes = adata_dict

    with settings.override(use_dask=usedask):
        dataset = MofaFlexDataset(adata_dict)
        filter_constant_features(dataset)
        result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]

    if arrays_are == "groups":
        arr = np.concatenate(
            [
                dataset.align_local_array_to_global(group["view1"], group_name, "view1", align_to="features", axis=1)
                for group_name, group in result.items()
            ],
            axis=0,
        )
        if arr.shape[1] > 0:
            assert not np.allclose(np.nanvar(arr, axis=0), 0)
        else:
            assert np.nanvar(arr, axis=0).shape[0] == 0

        constgenes = reduce(lambda x, y: x.intersection(y), constgenes.values())
        assert np.all(~constgenes.isin(dataset.feature_names["view1"]))

        allfeatures = adata_dict["group1"]["view1"].var_names.union(adata_dict["group2"]["view1"].var_names)
        assert arr.shape[1] == allfeatures.size - constgenes.size
    else:
        for group_name, group in result.items():
            for view_name, view in group.items():
                if view.shape[1] > 0:
                    assert not np.allclose(np.nanvar(view, axis=0), 0)
                else:
                    assert np.nanvar(view, axis=0).shape[0] == 0
                assert np.all(~constgenes[view_name].isin(dataset.feature_names[view_name]))
                assert view.shape[1] == adata_dict[group_name][view_name].n_vars - constgenes[view_name].size
