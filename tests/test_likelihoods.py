import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix
from scipy.special import expit, logit
from scipy.stats import bernoulli

from mofaflex import settings
from mofaflex._core import MofaFlexDataset
from mofaflex._core.likelihoods import Bernoulli, Likelihood, NegativeBinomial, Normal
from mofaflex._core.likelihoods.base import R2, LogLikelihoods
from mofaflex._core.utils import MeanStd, nanmin, sample_all_data_as_one_batch

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

    def test_logpmf(self, rng):
        logits = rng.standard_normal(size=1000)
        targets = rng.binomial(1, 0.5, size=1000)
        np.testing.assert_allclose(bernoulli.logpmf(targets, expit(logits)), Bernoulli._logpmf(targets, logits))


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


_group_name = "group_0"
_view_name = "view_0"


class TestDevianceExplained:
    """Fixed points of the fraction of deviance explained for nonnegative views."""

    @pytest.fixture(scope="class", params=[False, True], ids=["nonan", "withnan"])
    def y_true(self, rng, likelihood, request):
        match likelihood:
            case "Normal":
                arr = np.abs(rng.standard_normal(size=(50, 10)))
            case "Bernoulli":
                arr = rng.binomial(1, 0.3, size=(50, 10)).astype(np.float64)
            case "NegativeBinomial":
                # low mean, so that most features have a zero minimum, exercising the epsilon-regularized null model
                arr = rng.negative_binomial(0.5, 0.3, size=(50, 10)).astype(np.float64)
        arr[0, 0] = 0
        if request.param:
            arr[rng.random(arr.shape) < 0.1] = np.nan
        return arr

    @pytest.fixture(scope="class")
    def likelihood_obj(self, likelihood, y_true, create_adata):
        dataset = MofaFlexDataset({_group_name: {_view_name: create_adata(y_true)}}, cast_to=None)
        obj = Likelihood.known_likelihoods()[likelihood](_view_name, dataset, True)
        if isinstance(obj, NegativeBinomial):
            obj._dispersion = MeanStd(np.full(y_true.shape[1], 0.1), None)  # not trained, so set manually
        return obj

    @staticmethod
    def _raw_prediction(obj, target):
        """Invert transform_prediction: the raw linear predictor which predicts `target` on the scale of the data."""
        shift = obj._shift[_group_name]
        match obj:
            case Normal():
                return (target - shift) / obj._scale[_group_name]
            case Bernoulli():
                return logit(np.clip(target, settings.eps, 1 - settings.eps)) - shift
            case NegativeBinomial():
                return target / obj._sample_means[_group_name] - shift

    @staticmethod
    def _null_target(obj, y_true):
        """The prediction of the null model, on the scale of the data."""
        match obj:
            case Normal():
                return np.broadcast_to(obj._shift[_group_name], y_true.shape)
            case Bernoulli():
                null_probability = np.clip(nanmin(y_true, axis=0), settings.eps, 1 - settings.eps)
                return np.broadcast_to(null_probability, y_true.shape)
            case NegativeBinomial():
                sample_means = obj._sample_means[_group_name]
                return np.maximum(obj._shift[_group_name], settings.eps) * sample_means

    @staticmethod
    def _unclamped(obj, y_true, y_pred):
        """The fraction of deviance explained without the clamp to nonnegative values applied by Likelihood.r2."""
        res = obj._deviance_explained_impl(y_true, y_pred, _group_name)
        if isinstance(res, LogLikelihoods):
            res = R2(res.saturated - res.model, res.saturated - res.null)
        return 1.0 - res.ss_res / res.ss_tot

    def test_null_model(self, likelihood_obj, y_true):
        y_pred = self._raw_prediction(likelihood_obj, self._null_target(likelihood_obj, y_true))
        # the clamp in Likelihood.r2 would hide a negative value
        assert self._unclamped(likelihood_obj, y_true, y_pred) == pytest.approx(0, abs=1e-6)

    def test_saturated_model(self, likelihood_obj, y_true):
        y_pred = self._raw_prediction(likelihood_obj, y_true)
        assert likelihood_obj.r2(y_true, y_pred, _group_name) == pytest.approx(1, abs=1e-6)

    def test_intermediate_model(self, likelihood_obj, y_true):
        # halfway between the null and saturated models should score strictly between 0 and 1.
        target = (self._null_target(likelihood_obj, y_true) + y_true) / 2
        r2 = likelihood_obj.r2(y_true, self._raw_prediction(likelihood_obj, target), _group_name)
        assert 0.05 < r2 < 1.0

    @pytest.mark.parametrize(
        "res", [R2(np.nan, 1.0), R2(1.0, np.nan), R2(0.0, 0.0)], ids=["nan_res", "nan_tot", "zero_deviance"]
    )
    def test_degenerate_is_nan(self, likelihood_obj, y_true, res, monkeypatch):
        # a degenerate null model must not be turned into a plausible-looking 0 by the clamp
        monkeypatch.setattr(likelihood_obj, "_deviance_explained_impl", lambda *args, **kwargs: res)
        assert np.isnan(likelihood_obj.r2(y_true, np.zeros_like(y_true), _group_name))

    @pytest.mark.parametrize("res", [R2(np.inf, 1.0), R2(1.0, 0.0)], ids=["inf_res", "zero_tot"])
    def test_degenerate_clamps(self, likelihood_obj, y_true, res, monkeypatch):
        # the model is infinitely worse than the null model, e.g. a count of zero predicted for a nonzero observation
        monkeypatch.setattr(likelihood_obj, "_deviance_explained_impl", lambda *args, **kwargs: res)
        assert likelihood_obj.r2(y_true, np.zeros_like(y_true), _group_name) == 0
