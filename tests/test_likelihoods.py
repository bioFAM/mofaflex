import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix, issparse
from scipy.special import expit
from scipy.stats import bernoulli

from mofaflex._core import MofaFlexDataset
from mofaflex._core.likelihoods import Bernoulli, Likelihood, NegativeBinomial, Normal
from mofaflex._core.likelihoods.base import R2, LogLikelihoods
from mofaflex._core.utils import MeanStd, sample_all_data_as_one_batch

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


class _TestLikelihood:
    @pytest.fixture(scope="class")
    def y_true(self, dataset):
        return dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]

    def test_r2(self, y_true, likelihoods, nonnegative, subtests):
        for group_name, group in y_true.items():
            for view_name, view in group.items():
                if issparse(view):
                    view = view.toarray()
                likelihood = likelihoods[view_name]

                with subtests.test("null model"):
                    null = np.broadcast_to(0, view.shape)
                    if nonnegative[view_name]:
                        res = likelihood._deviance_explained_impl(view, null, group_name)
                    else:
                        res = likelihood._r2_impl(view, likelihood.transform_prediction(null, group_name), group_name)
                    if isinstance(res, LogLikelihoods):
                        res = R2(res.saturated - res.model, res.saturated - res.null)

                    # the clamp in Likelihood.r2 would hide a negative value
                    assert np.isclose(0, 1.0 - res.ss_res / res.ss_tot)

                with subtests.test("saturated model"):
                    r2 = likelihood.r2(view, likelihood.transform_data(view, group_name), group_name)
                    assert np.isclose(1, r2)

                with subtests.test("intermediate model"):
                    r2 = likelihood.r2(view, likelihood.transform_data(0.5 * (null + view), group_name), group_name)
                    assert 0.05 < r2 < 1

    @pytest.mark.parametrize(
        "r2", (R2(np.nan, 1.0), R2(1.0, np.nan), R2(0.0, 0.0)), ids=("nan_res", "nan_tot", "zero_deviance")
    )
    def test_degenerate_is_nan(self, likelihoods, y_true, r2, monkeypatch):
        # a degenerate null model must not be turned into a plausible-looking 0 by the clamp
        for view_name, likelihood in likelihoods.items():
            func = lambda *args, **kwargs: r2
            monkeypatch.setattr(likelihood, "_deviance_explained_impl", func)
            monkeypatch.setattr(likelihood, "_r2_impl", func)
            for group_name, group in y_true.items():
                view = group[view_name]
                if issparse(view):
                    view = view.toarray()
                assert np.isnan(likelihood.r2(view, np.broadcast_to(0, view.shape), group_name))

    @pytest.mark.parametrize("r2", (R2(np.inf, 1.0), R2(1.0, 0.0)), ids=("inf_res", "zero_tot"))
    def test_degenerate_clamps(self, likelihoods, y_true, r2, monkeypatch):
        # the model is infinitely worse than the null model, e.g. a count of zero predicted for a nonzero observation
        for view_name, likelihood in likelihoods.items():
            func = lambda *args, **kwargs: r2
            monkeypatch.setattr(likelihood, "_deviance_explained_impl", func)
            monkeypatch.setattr(likelihood, "_r2_impl", func)
            for group_name, group in y_true.items():
                view = group[view_name]
                if issparse(view):
                    view = view.toarray()
                assert likelihood.r2(view, np.broadcast_to(0, view.shape), group_name) == 0


class TestNormal(_TestLikelihood):
    @pytest.fixture(scope="class", params=[True, False])
    def scale_per_group(self, request):
        return request.param

    @pytest.fixture(scope="class")
    def likelihoods(self, dataset, nonnegative, scale_per_group):
        return {view_name: Normal(view_name, dataset, nn, scale_per_group) for view_name, nn in nonnegative.items()}

    @pytest.fixture(scope="class")
    def dataset(self, adata_dict):
        return MofaFlexDataset(adata_dict("Normal"), cast_to=None)

    def test_center_data(self, likelihoods, y_true, nonnegative):
        for group_name, group in y_true.items():
            for view_name, view in group.items():
                if nonnegative[view_name]:
                    assert np.allclose(np.nanmin(view - likelihoods[view_name]._shift[group_name], axis=0), 0)
                else:
                    assert np.allclose((view - likelihoods[view_name]._shift[group_name]).mean(axis=0), 0)

    def test_scale_data(self, likelihoods, dataset, y_true, scale_per_group):
        if scale_per_group:
            for group_name, group in y_true.items():
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
                        for group_name, group in y_true.items()
                        if view_name in group
                    ],
                    axis=0,
                )
                assert np.allclose((concat / likelihoods[view_name]._scale).var(), 1)


class TestBernoulli(_TestLikelihood):
    @pytest.fixture(scope="class")
    def likelihoods(self, dataset, nonnegative):
        return {view_name: Bernoulli(view_name, dataset, nn) for view_name, nn in nonnegative.items()}

    @pytest.fixture(scope="class")
    def dataset(self, adata_dict, sparse_arr):
        return MofaFlexDataset(adata_dict("Bernoulli", sparse_arr), cast_to=None)

    def test_center_data(self, likelihoods, y_true, nonnegative):
        for group_name, group in y_true.items():
            for view_name, view in group.items():
                assert np.allclose(np.nanmean(view - expit(likelihoods[view_name]._shift[group_name]), axis=0), 0)

    def test_logpmf(self, rng):
        logits = rng.standard_normal(size=1000)
        targets = rng.binomial(1, 0.5, size=1000)
        np.testing.assert_allclose(bernoulli.logpmf(targets, expit(logits)), Bernoulli._logpmf(targets, logits))


class TestNegativeBinomial(_TestLikelihood):
    @pytest.fixture(scope="class")
    def likelihoods(self, dataset, nonnegative):
        ret = {}
        for view_name, nn in nonnegative.items():
            likelihood = NegativeBinomial(view_name, dataset, nn)
            likelihood._dispersion = MeanStd(
                np.full(dataset.n_features[view_name], 0.1), None
            )  # not trained, so set manually
            ret[view_name] = likelihood
        return ret

    @pytest.fixture(scope="class")
    def dataset(self, adata_dict, sparse_arr):
        return MofaFlexDataset(adata_dict("NegativeBinomial", sparse_arr), cast_to=None)

    def test_center_data(self, likelihoods, y_true, nonnegative):
        for group_name, group in y_true.items():
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
