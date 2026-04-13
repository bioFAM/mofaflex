"""Tests for mofaflex.tl.consensus."""
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from mofaflex import terms
from mofaflex.tl import ConsensusResult, DataGenerator, KSelectionResult, fit_consensus, k_selection


@pytest.fixture(scope="module")
def consensus_mdata():
    """Small, fully non-negative, 2-view synthetic dataset as a nested AnnData dict."""
    gen = DataGenerator(
        n_features=[25, 30],
        n_samples=120,
        likelihoods=["Normal", "Normal"],
        n_fully_shared_factors=3,
        n_partially_shared_factors=1,
        n_private_factors=1,
        factor_size_params=(0.4, 0.6),
        n_active_factors=1.0,
        nmf=[True, True],
    )
    gen.generate(np.random.default_rng(0))

    # Build a single-group nested dict of AnnData (bypasses the MuData path).
    ys = gen._ys
    obs_names = [f"cell_{i}" for i in range(gen.n_samples)]
    data = {"group_1": {}}
    for m in range(gen.n_views):
        adata = ad.AnnData(
            ys[m].astype(np.float32, copy=False),
            obs=pd.DataFrame(index=obs_names),
            var=pd.DataFrame(index=[f"view{m}_feature_{j}" for j in range(gen.n_features[m])]),
        )
        data["group_1"][f"view_{m}"] = adata
    return data


def _nonneg_template_factory(n_factors=5):
    def factory():
        return terms.MofaFlex(
            n_factors=n_factors,
            nonnegative_factors=True,
            nonnegative_weights=True,
        )

    return factory


@pytest.fixture(scope="module")
def consensus_result(consensus_mdata):
    return fit_consensus(
        _nonneg_template_factory(n_factors=5),
        consensus_mdata,
        n_runs=3,
        density_threshold=2.0,  # permissive; tiny n_runs gives higher densities
        seed=123,
        fit_kwargs={"max_epochs": 50, "device": "cpu", "lr": 0.01, "early_stopper_patience": 20},
        show_progress=False,
    )


def test_consensus_result_shapes(consensus_result, consensus_mdata):
    res = consensus_result
    assert isinstance(res, ConsensusResult)
    assert res.n_factors == 5
    assert set(res.view_names) == {"view_0", "view_1"}
    assert len(res.per_run_seeds) == 3
    assert len(set(res.per_run_seeds)) == 3

    for view in res.view_names:
        df = res.consensus_weights[view]
        assert df.shape == (consensus_mdata["group_1"][view].n_vars, 5)
        assert (df.values >= 0).all()

    for group in res.group_names:
        df = res.consensus_factors[group]
        assert df.shape[1] == 5
        assert (df.values >= 0).all()


def test_consensus_stability_and_error(consensus_result):
    # Stability is silhouette score in [-1, 1]; reconstruction error must be finite.
    assert -1.0 <= consensus_result.stability <= 1.0
    assert np.isfinite(consensus_result.reconstruction_error)
    assert consensus_result.reconstruction_error >= 0


def test_consensus_diagnostics_shapes(consensus_result):
    res = consensus_result
    n_spectra = 3 * 5
    assert len(res.local_density) == n_spectra
    assert len(res.kept_mask) == n_spectra
    assert res.kept_mask.sum() >= res.n_factors
    assert len(res.cluster_labels) == int(res.kept_mask.sum())
    assert set(res.cluster_labels.unique()).issubset(set(range(res.n_factors)))


def test_fit_consensus_rejects_signed_template(consensus_mdata):
    def factory():
        return terms.MofaFlex(
            n_factors=4,
            nonnegative_factors=False,
            nonnegative_weights=False,
        )

    with pytest.raises(ValueError, match="non-negative"):
        fit_consensus(
            factory,
            consensus_mdata,
            n_runs=2,
            seed=0,
            fit_kwargs={"max_epochs": 20, "device": "cpu"},
            show_progress=False,
        )


def test_fit_consensus_requires_n_runs_at_least_2(consensus_mdata):
    with pytest.raises(ValueError, match="n_runs"):
        fit_consensus(
            _nonneg_template_factory(n_factors=3),
            consensus_mdata,
            n_runs=1,
            seed=0,
            fit_kwargs={"max_epochs": 10, "device": "cpu"},
            show_progress=False,
        )


def test_consensus_factors_sorted_by_usage(consensus_result):
    # cnmf.py:950-957: factors are reordered by total usage contribution
    # descending. Verify the per-group column-sum totals are non-increasing.
    totals = None
    for Z in consensus_result.consensus_factors.values():
        col_sums = Z.values.sum(axis=0)
        totals = col_sums if totals is None else totals + col_sums
    assert np.all(np.diff(totals) <= 1e-9), f"factors not sorted by usage: {totals}"


def test_k_selection_sweep(consensus_mdata):
    def factory(k):
        return terms.MofaFlex(
            n_factors=k,
            nonnegative_factors=True,
            nonnegative_weights=True,
        )

    result = k_selection(
        factory,
        consensus_mdata,
        k_values=[3, 5],
        n_runs=2,
        density_threshold=2.0,
        seed=7,
        fit_kwargs={"max_epochs": 50, "device": "cpu", "lr": 0.01, "early_stopper_patience": 20},
        show_progress=False,
    )
    assert isinstance(result, KSelectionResult)
    assert set(result.results.keys()) == {3, 5}
    assert list(result.stability.index) == [3, 5]
    assert list(result.reconstruction_error.index) == [3, 5]
    for k, r in result.results.items():
        assert r.n_factors == k

    # Plotting should run without error and return a plotnine ggplot.
    import plotnine as p9

    plot = result.plot()
    assert isinstance(plot, p9.ggplot)
