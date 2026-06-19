import warnings

import numpy as np
import pytest
from mudata import MuData

from mofaflex.tl import DataGenerator


@pytest.fixture(scope="module")
def generator_kwargs():
    return {
        "n_features": [20, 30, 25],
        "n_samples": 1000,
        "likelihoods": ["Normal", "Bernoulli", "Poisson"],
        "n_fully_shared_factors": 2,
        "n_partially_shared_factors": 15,
        "n_active_factors": 1,
        "factor_size_params": (0.3, 0.6),
        "n_private_factors": 3,
        "nmf": [False, False, True],
    }


@pytest.fixture
def generator(rng, generator_kwargs):
    gen = DataGenerator(**generator_kwargs)
    gen.generate(rng)
    return gen


@pytest.fixture
def generator_with_missing(rng, generator):
    generator.generate_missingness(
        rng, n_partial_samples=10, n_partial_features=13, missing_fraction_partial_features=0.01, random_fraction=0.01
    )
    return generator


@pytest.fixture
def view_feature_indices(generator_kwargs):
    idx = np.concatenate(([0], np.cumsum(generator_kwargs["n_features"])))
    return idx[:-1], idx[1:]


def test_attributes(generator, generator_kwargs):
    assert generator.n_samples == generator_kwargs["n_samples"]
    assert generator.n_features == generator_kwargs["n_features"]
    assert generator.likelihoods == generator_kwargs["likelihoods"]
    assert generator.n_views == len(generator_kwargs["n_features"])
    assert generator.n_fully_shared_factors == generator_kwargs["n_fully_shared_factors"]
    assert generator.n_partially_shared_factors == generator_kwargs["n_partially_shared_factors"]
    assert generator.n_private_factors == generator_kwargs["n_private_factors"]
    assert generator.n_active_factors == generator_kwargs["n_active_factors"]
    assert generator.nmf == generator_kwargs["nmf"]

    assert (
        generator.n_factors
        == generator.n_fully_shared_factors + generator.n_partially_shared_factors + generator.n_private_factors
    )


def test_generate(generator, view_feature_indices):
    assert generator.y.shape == (generator.n_samples, sum(generator.n_features))
    for startidx, endidx, likelihood, nmf in zip(
        *view_feature_indices, generator.likelihoods, generator.nmf, strict=False
    ):
        match likelihood:
            case "Bernoulli":
                assert np.all(
                    np.isclose(generator.y[:, startidx:endidx], 0) | np.isclose(generator.y[:, startidx:endidx], 1)
                )
            case "Poisson":
                assert np.min(generator.y[:, startidx:endidx]) >= 0
                assert np.allclose(generator.y[:, startidx:endidx], np.round(generator.y[:, startidx:endidx]))
        if nmf:
            assert np.all(generator.w[:, startidx:endidx] >= 0)

    assert generator.w.shape == (generator.n_factors, np.sum(generator.n_features))

    assert np.sum(np.all(~generator.w_mask, axis=1)) == generator.n_factors - int(
        generator.n_active_factors * generator.n_factors
    )

    assert generator.z.shape == (generator.n_samples, generator.n_factors)

    if any(generator.nmf):
        assert np.all(generator.z >= 0)

    assert np.all((np.abs(generator.w) >= 0.1) == generator.w_mask)

    active_factors = np.zeros(generator.n_factors, dtype=int)
    for startidx, endidx in zip(*view_feature_indices, strict=False):
        active_factors += np.any(generator.w_mask[:, startidx:endidx], axis=1)
    assert np.sum(active_factors == generator.n_views) == generator.n_fully_shared_factors
    assert np.sum((active_factors > 1) & (active_factors < generator.n_views)) == generator.n_partially_shared_factors
    assert np.sum(active_factors == 1) == generator.n_private_factors


def test_normalize(generator, view_feature_indices):
    generator.normalize(with_std=True)

    for (i, startidx), endidx in zip(enumerate(view_feature_indices[0]), view_feature_indices[1], strict=False):
        if generator.likelihoods[i] == "Normal":
            assert np.allclose(generator.y[:, startidx:endidx].mean(axis=0), 0)
            assert np.allclose(generator.y[:, startidx:endidx].var(axis=0), 1)


def test_missingness(rng, generator, view_feature_indices):
    generator.generate_missingness(rng, random_fraction=0.01)
    assert np.isnan(generator.missing_y).sum() == int(0.01 * np.prod(generator.missing_y.shape))

    generator.generate_missingness(rng, n_partial_samples=10)
    missing_samples = np.zeros(generator.n_samples, dtype=bool)
    for (i, startidx), endidx in zip(enumerate(view_feature_indices[0]), view_feature_indices[1], strict=False):
        missing_samples |= np.isnan(generator.missing_y[:, startidx:endidx]).sum(axis=1) == generator.n_features[i]
    assert missing_samples.sum() == 10

    generator.generate_missingness(rng, n_partial_features=13, missing_fraction_partial_features=0.01)
    missing_features = np.isnan(generator.missing_y).sum(axis=0) == int(0.01 * generator.n_samples)
    assert missing_features.sum() == 13


def test_permutation(rng, generator_with_missing, view_feature_indices):
    factor_permutation = rng.permutation(generator_with_missing.n_factors)
    oldz = generator_with_missing.z
    generator_with_missing.permute_factors(factor_permutation)
    assert np.all(generator_with_missing.z == oldz[:, factor_permutation])

    feature_permutation = [rng.permutation(n_features) for n_features in generator_with_missing.n_features]
    oldw = generator_with_missing.w
    oldy = generator_with_missing.y
    oldmissingy = generator_with_missing.missing_y

    for startidx, endidx, permutation in zip(*view_feature_indices, feature_permutation, strict=False):
        oldw[:, startidx:endidx] = oldw[:, startidx:endidx][:, permutation]
        oldy[:, startidx:endidx] = oldy[:, startidx:endidx][:, permutation]
        oldmissingy[:, startidx:endidx] = oldmissingy[:, startidx:endidx][:, permutation]

    generator_with_missing.permute_features(feature_permutation)
    assert np.all(generator_with_missing.w == oldw)

    assert np.all(generator_with_missing.y == oldy)
    assert np.all(np.isnan(generator_with_missing.missing_y) == np.isnan(oldmissingy))
    assert np.all(
        generator_with_missing.missing_y[~np.isnan(generator_with_missing.missing_y)]
        == oldmissingy[~np.isnan(oldmissingy)]
    )


def test_mudata(generator_with_missing):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        mdata = generator_with_missing.to_mudata()
    assert isinstance(mdata, MuData)
    assert mdata.uns["n_active_factors"] == generator_with_missing.n_active_factors
    assert np.all(mdata.obsm["z"] == generator_with_missing.z)

    lastidx = 0
    for view_idx in range(generator_with_missing.n_views):
        view_name = f"feature_group_{view_idx}"
        assert mdata.uns["likelihoods"][view_name] == generator_with_missing.likelihoods[view_idx]
        adata = mdata[view_name]
        newidx = lastidx + generator_with_missing.n_features[view_idx]
        assert np.all(adata.varm["w"].T == generator_with_missing.w[:, lastidx:newidx])

        y = generator_with_missing.missing_y[:, lastidx:newidx]
        assert np.allclose(adata.X, y, equal_nan=True)
        lastidx = newidx
