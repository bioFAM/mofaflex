import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix
from scipy.special import expit

from mofaflex._core import MofaFlexDataset
from mofaflex._core.likelihoods import Bernoulli, Likelihood, NegativeBinomial, Normal
from mofaflex._core.utils import sample_all_data_as_one_batch

_sparse_arr = [csc_array, csc_matrix, csr_array, csr_matrix]

_ngroups = 2
_nviews = 3


@pytest.fixture(scope="module", params=["Normal", "Bernoulli", "NegativeBinomial"])
def likelihood(request):
    return request.param


@pytest.fixture(scope="module", params=[np.asarray, csc_array, csc_matrix, csr_array, csr_matrix])
def adata(rng, create_adata, likelihood, random_array, request):
    return create_adata(request.param(random_array(likelihood, (20, 5))))


def test_infer_likelihoods(adata, likelihood):
    inferred = Likelihood.infer(adata)
    assert likelihood == inferred.__name__


def test_validate_likelihoods(adata, likelihood):
    Likelihood.known_likelihoods()[likelihood].validate(adata, None, None)


@pytest.fixture(scope="module")
def group_names():
    return [f"group_{group}" for group in range(_ngroups)]


@pytest.fixture(scope="module")
def view_names():
    return [f"view_{view}" for view in range(_nviews)]


@pytest.fixture(scope="module")
def adata_dict(rng, create_adata, random_array, group_names, view_names):
    def generate(likelihood, sparse_arr=None):
        data = {}
        for group_name in group_names:
            cdata = {}
            for view_name in view_names:
                arr = random_array(likelihood, (100, 30))
                if sparse_arr is not None:
                    arr = sparse_arr[group_name][view_name](arr)
                cdata[view_name] = create_adata(arr, obs_names=[f"{group_name}_{i}" for i in range(arr.shape[0])])
            data[group_name] = cdata
        return data

    return generate


@pytest.fixture(scope="module", params=[True, False])
def nonnegative(view_names, request):
    return dict.fromkeys(view_names, request.param)


@pytest.fixture(scope="module")
def sparse_arr(group_names, view_names):
    i = 0
    fundict = {}
    for group_name in group_names:
        cdict = {}
        for view_name in view_names:
            cdict[view_name] = _sparse_arr[i % len(_sparse_arr)]
            i += 1
        fundict[group_name] = cdict
    return fundict


class TestNormal:
    @pytest.fixture(scope="class", params=[True, False])
    def scale_per_group(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def likelihoods(self, dataset, nonnegative, scale_per_group):
        return {view_name: Normal(view_name, dataset, nn, scale_per_group) for view_name, nn in nonnegative.items()}

    @pytest.fixture(scope="class")
    def dataset(self, adata_dict):
        return MofaFlexDataset(adata_dict("Normal"), cast_to=None)

    def test_center_data(self, likelihoods, dataset, nonnegative):
        result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]
        for group_name, group in result.items():
            for view_name, view in group.items():
                if nonnegative[view_name]:
                    assert np.allclose(np.nanmin(view - likelihoods[view_name]._shift[group_name], axis=0), 0)
                else:
                    assert np.allclose((view - likelihoods[view_name]._shift[group_name]).mean(axis=0), 0)

    def test_scale_data(self, likelihoods, dataset, scale_per_group):
        result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]
        if scale_per_group:
            for group_name, group in result.items():
                for view_name, view in group.items():
                    assert np.allclose(
                        (
                            (view - likelihoods[view_name]._shift[group_name])
                            / likelihoods[view_name]._scale[group_name]
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


class TestBernoulli:
    @pytest.fixture(scope="class")
    def likelihoods(self, dataset, nonnegative):
        return {view_name: Bernoulli(view_name, dataset, nn) for view_name, nn in nonnegative.items()}

    @pytest.fixture(scope="class")
    def dataset(self, adata_dict, sparse_arr):
        return MofaFlexDataset(adata_dict("Bernoulli", sparse_arr), cast_to=None)

    def test_center_data(self, likelihoods, dataset, nonnegative):
        result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]
        for group_name, group in result.items():
            for view_name, view in group.items():
                assert np.allclose(np.nanmean(view - expit(likelihoods[view_name]._shift[group_name]), axis=0), 0)


class TestNegativeBinomial:
    @pytest.fixture(scope="class")
    def likelihoods(self, dataset, nonnegative):
        return {view_name: NegativeBinomial(view_name, dataset, nn) for view_name, nn in nonnegative.items()}

    @pytest.fixture(scope="class")
    def dataset(self, adata_dict, sparse_arr):
        return MofaFlexDataset(adata_dict("NegativeBinomial", sparse_arr), cast_to=None)

    def test_center_data(self, likelihoods, dataset, nonnegative):
        result = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]
        for group_name, group in result.items():
            for view_name, view in group.items():
                if nonnegative[view_name]:
                    assert np.allclose(np.nanmin(view - likelihoods[view_name]._shift[group_name], axis=0), 0)
                else:
                    assert np.allclose(
                        (
                            view / likelihoods[view_name]._sample_means[group_name]
                            - likelihoods[view_name]._shift[group_name]
                        ).mean(axis=0),
                        0,
                    )
