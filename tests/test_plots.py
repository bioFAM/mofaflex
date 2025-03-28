import warnings
from functools import partial, wraps
from pathlib import Path

import matplotlib.pyplot as plt
import mudata as md
import plotnine
import pytest
from matplotlib.testing.decorators import image_comparison as mpl_image_comparison

import prismo as pr

image_comparison = partial(
    mpl_image_comparison, extensions=["png"], tol=0.5
)  # tolerance for differences in text rendering
plotnine.options.base_family = "DejaVu Sans"  # bundled with Matplotlib


@pytest.fixture(scope="module")
def cll_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        return md.read_h5mu(Path(__file__).parent / "plots" / "cll.h5mu")


@pytest.fixture(scope="module")
def cll_model():
    return pr.PRISMO.load(Path(__file__).parent / "plots" / "cll_model.h5")


@pytest.fixture(scope="module")
def mousebrain_model():
    return pr.PRISMO.load(Path(__file__).parent / "plots" / "mousebrain_model.h5")


def wrap_plotnine(func):
    @wraps(func)
    def decorated(*args, **kwargs):
        plots = func(*args, **kwargs)
        if not isinstance(plots, list) and not isinstance(plots, tuple):
            plots = [plots]
        for plot in plots:
            plt.figure(plot.draw(show=False))

    return decorated


@image_comparison(baseline_images=["overview"])
@wrap_plotnine
def test_overview(cll_data):
    return pr.pl.overview(cll_data)


@image_comparison(baseline_images=["training_curve"])
@wrap_plotnine
def test_training_curve(cll_model):
    return pr.pl.training_curve(cll_model)


@image_comparison(baseline_images=["factor_correlation"])
@wrap_plotnine
def test_factor_correlation(cll_model):
    return pr.pl.factor_correlation(cll_model)


@image_comparison(baseline_images=["variance_explained"])
@wrap_plotnine
def test_variance_explained(cll_model):
    return pr.pl.variance_explained(cll_model)


@image_comparison(baseline_images=["all_weights", "all_weights_Mutations", "all_weights_Mutations_mRNA"])
@wrap_plotnine
def test_all_weights(cll_model):
    return (
        pr.pl.all_weights(cll_model),
        pr.pl.all_weights(cll_model, views="Mutations"),
        pr.pl.all_weights(cll_model, views=["Mutations", "mRNA"]),
    )


@image_comparison(baseline_images=["factor"])
@wrap_plotnine
def test_factor(cll_model):
    return pr.pl.factor(cll_model)


@image_comparison(
    baseline_images=[
        "top_weights",
        "top_weights_features-5",
        "top_weights_view-Mutations",
        "top_weights_views-Mutations-mRNA",
        "top_weights_factor-1",
        "top_weights_factors-1-7",
        "top_weights_view-Mutations_factor_1",
        "top_weights_nrows-2",
    ]
)
@wrap_plotnine
def test_top_weights(cll_model):
    return (
        pr.pl.top_weights(cll_model, figsize=(20, 20)),
        pr.pl.top_weights(cll_model, n_features=5, figsize=(20, 20)),
        pr.pl.top_weights(cll_model, views="Mutations", figsize=(20, 20)),
        pr.pl.top_weights(cll_model, views=["Mutations", "mRNA"], figsize=(20, 20)),
        pr.pl.top_weights(cll_model, factors=1),
        pr.pl.top_weights(cll_model, factors=["Factor 1", "Factor 7"]),
        pr.pl.top_weights(cll_model, views="Mutations", factors=1),
        pr.pl.top_weights(cll_model, nrow=2, figsize=(20, 5)),
    )


@image_comparison(
    baseline_images=[
        "weights",
        "weights_features-5",
        "weights_view-Mutations",
        "weights_views-Mutations-mRNA",
        "weights_factor-1",
        "weights_factors-1-7",
        "weights_view-Mutations_factor_1",
        "weights_views-Mutations-mRNA_nrows-3",
    ]
)
@wrap_plotnine
def test_weights(cll_model):
    return (
        pr.pl.weights(cll_model, figsize=(40, 20)),
        pr.pl.weights(cll_model, n_features=5, figsize=(40, 20)),
        pr.pl.weights(cll_model, views="Mutations", figsize=(40, 5)),
        pr.pl.weights(cll_model, views=["Mutations", "mRNA"], figsize=(40, 10)),
        pr.pl.weights(cll_model, factors=1),
        pr.pl.weights(cll_model, factors=["Factor 1", "Factor 7"]),
        pr.pl.weights(cll_model, views="Mutations", factors=1),
        pr.pl.weights(cll_model, views=["Mutations", "mRNA"], nrow=3, figsize=(30, 15)),
    )


@image_comparison(baseline_images=["weight_sparsity_histogram"])
@wrap_plotnine
def test_weight_sparsity_histogram(cll_model):
    return pr.pl.weight_sparsity_histogram(cll_model)


@image_comparison(baseline_images=["top_weights_annotations"])
@wrap_plotnine
def test_top_weights_annotations(mousebrain_model):
    return pr.pl.top_weights(mousebrain_model, figsize=(20, 20))


@image_comparison(baseline_images=["weights_annotations"])
@wrap_plotnine
def test_weights_annotations(mousebrain_model):
    return pr.pl.weights(mousebrain_model, factors=["Factor 1", "Factor 2", "Astrocytes", "Interneurons"])


@image_comparison(baseline_images=["factors_scatter", "factors_scatter-color"])
@wrap_plotnine
def test_factors_scatter(mousebrain_model):
    return pr.pl.factors_scatter(mousebrain_model, 1, "Astrocytes"), pr.pl.factors_scatter(
        mousebrain_model, 1, "Astrocytes", color="log1p_total_counts"
    )


@image_comparison(
    baseline_images=[
        "covariates_factor_scatter",
        "covariates_factor_scatter_Astrocytes",
        "covariates_factor_scatter_cov0_color-Astrocytes",
        "covariates_factor_scatter_cov0_color-log1p_total_counts",
        "covariates_factor_scatter_cov1-cov0",
    ]
)
@wrap_plotnine
def test_covariates_factor_scatter(mousebrain_model):
    return (
        pr.pl.covariates_factor_scatter(mousebrain_model, 1),
        pr.pl.covariates_factor_scatter(mousebrain_model, "Astrocytes"),
        pr.pl.covariates_factor_scatter(mousebrain_model, 1, covariate_dims=0, color="Astrocytes"),
        pr.pl.covariates_factor_scatter(mousebrain_model, 1, covariate_dims=0, color="log1p_total_counts"),
        pr.pl.covariates_factor_scatter(mousebrain_model, 1, covariate_dims=(1, 0)),
    )


@image_comparison(baseline_images=["factors_covariate-cov0", "factors_covariate-cov1-cov0"])
@wrap_plotnine
def test_factors_covariate(mousebrain_model):
    return pr.pl.factors_covariate(mousebrain_model, 0, figsize=(60, 4)), pr.pl.factors_covariate(
        mousebrain_model, 1, 0, figsize=(60, 4)
    )


@image_comparison(baseline_images=["gp_covariate"])
@wrap_plotnine
def test_gp_covariate(mousebrain_model):
    return pr.pl.gp_covariate(mousebrain_model, size=0.25, figsize=(60, 4))


@image_comparison(baseline_images=["smoothness"])
@wrap_plotnine
def test_smoothness(mousebrain_model):
    return pr.pl.smoothness(mousebrain_model, figsize=(5, 5))
