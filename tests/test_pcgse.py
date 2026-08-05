import pandas as pd
import pytest

import mofaflex as mfl
from mofaflex._core.pcgse import _test_single_view, pcgse_test


def test_test_annotation_significance_data_None_corr_True(mousebrain_model):
    with pytest.raises(ValueError):
        mfl.tl.test_annotation_significance(
            mousebrain_model, mousebrain_model.get_weight_annotations(), data=None, corr_adjust=True
        )


def test_test_annotation_significance_annotations_empty(mousebrain_model):
    results = mfl.tl.test_annotation_significance(mousebrain_model, {}, corr_adjust=False)
    assert results == {}


def test_test_annotation_significance(mousebrain_model):
    results = mfl.tl.test_annotation_significance(
        mousebrain_model, mousebrain_model.get_weight_annotations(), data=None, corr_adjust=False
    )

    assert isinstance(results, dict)
    assert "view_1" in results

    for key in ("factor", "annotation", "t", "p", "padj", "sign"):
        assert key in results["view_1"]


def test_pcgse_test(mousebrain_model):
    results = pcgse_test(
        data=None,
        nonnegative_weights={"view_1": True},
        annotations=mousebrain_model.get_weight_annotations(),
        weights=mousebrain_model.get_weights(),
        corr_adjust=False,
    )

    assert isinstance(results, dict)
    assert "view_1" in results
    assert results["view_1"]["sign"].unique() == ["pos"]


def test_test_single_view_nonnegative(mousebrain_model):
    assert (
        _test_single_view(
            "view_1",
            nonnegative_weights=True,
            feature_sets=mousebrain_model.get_weight_annotations()["view_1"],
            factor_loadings=mousebrain_model.get_weights()["view_1"],
            y=None,
            sign="neg",
            corr_adjust=False,
        )
        is None
    )


def test_test_single_view_empty(mousebrain_model):
    feature_sets_empty = mousebrain_model.get_weight_annotations()["view_1"] & False

    assert (
        _test_single_view(
            "view_1",
            nonnegative_weights=True,
            feature_sets=feature_sets_empty,
            factor_loadings=mousebrain_model.get_weights()["view_1"],
            y=None,
            sign="pos",
            corr_adjust=False,
        )
        is None
    )


def test_test_single_view_repeated_factor_names(mousebrain_model):
    factors = mousebrain_model.get_weights()["view_1"]
    result = _test_single_view(
        "view_1",
        nonnegative_weights=True,
        feature_sets=mousebrain_model.get_weight_annotations()["view_1"],
        factor_loadings=pd.concat((factors, factors), axis=1),
        y=None,
        sign="pos",
        corr_adjust=False,
    )
    for df in result.values():
        assert df.shape[0] == factors.shape[1] * 2


@pytest.mark.parametrize("sign", ("pos", "neg", "all"))
def test_test_single_view(mousebrain_model, sign):
    results = _test_single_view(
        "view_1",
        nonnegative_weights=False,
        feature_sets=mousebrain_model.get_weight_annotations()["view_1"],
        factor_loadings=mousebrain_model.get_weights()["view_1"],
        y=None,
        sign=sign,
        corr_adjust=False,
    )

    assert isinstance(results, dict)
    assert "t" in results
    assert "p" in results
    assert "padj" in results


def test_single_guided_factor(mousebrain_model):
    results = _test_single_view(
        "view_1",
        nonnegative_weights=False,
        feature_sets=mousebrain_model.get_weight_annotations()["view_1"].iloc[:, :1],
        factor_loadings=mousebrain_model.get_weights()["view_1"],
        y=None,
        sign="all",
        corr_adjust=False,
    )
    for res in results.values():
        assert res.shape == (mousebrain_model.n_total_factors, 1)
