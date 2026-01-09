import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from mofaflex._core import MofaFlexDataset
from mofaflex._core.likelihoods import Normal
from mofaflex._core.utils import sample_all_data_as_one_batch

_sparse_arr = [csc_array, csc_matrix, csr_array, csr_matrix]

_ngroups = 2
_nviews = 3


@pytest.fixture(scope="module")
def group_names():
    return [f"group_{group}" for group in range(_ngroups)]


@pytest.fixture(scope="module")
def view_names():
    return [f"view_{view}" for view in range(_nviews)]


@pytest.fixture(scope="module")
def adata_dict(rng, create_adata, random_array, group_names, view_names):
    data = {}
    for group_name in group_names:
        cdata = {}
        for view_name in view_names:
            arr = random_array("Normal", (100, 30))
            cdata[view_name] = create_adata(arr, obs_names=[f"{group_name}_{i}" for i in range(arr.shape[0])])
        data[group_name] = cdata
    return data


@pytest.fixture(scope="module", params=[True, False])
def nonnegative(view_names, request):
    return dict.fromkeys(view_names, request.param)


@pytest.fixture(scope="module", params=[True, False])
def scale_per_group(request):
    return request.param


@pytest.fixture(scope="module")
def dataset(adata_dict):
    return MofaFlexDataset(adata_dict, cast_to=None)


@pytest.fixture(scope="module")
def likelihoods(dataset, nonnegative, scale_per_group):
    return {view_name: Normal(view_name, dataset, nn, scale_per_group) for view_name, nn in nonnegative.items()}


def test_center_data(likelihoods, dataset, nonnegative):
    result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]
    for group_name, group in result.items():
        for view_name, view in group.items():
            if nonnegative[view_name]:
                assert np.allclose(np.nanmin(view - likelihoods[view_name]._shift[group_name], axis=0), 0)
            else:
                assert np.allclose((view - likelihoods[view_name]._shift[group_name]).mean(axis=0), 0)


def test_scale_data(likelihoods, dataset, scale_per_group):
    result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]
    if scale_per_group:
        for group_name, group in result.items():
            for view_name, view in group.items():
                assert np.allclose(
                    (
                        (view - likelihoods[view_name]._shift[group_name]) / likelihoods[view_name]._scale[group_name]
                    ).var(),
                    1,
                )
    else:
        for view_name in dataset.view_names:
            concat = np.concat(
                [
                    group[view_name] - likelihoods[view_name]._shift[group_name]
                    for group_name, group in result.items()
                    if view_name in group
                ],
                axis=0,
            )
            assert np.allclose((concat / likelihoods[view_name]._scale).var(), 1)
