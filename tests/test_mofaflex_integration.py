# integration tests: only testing if the code runs without errors
import builtins
import warnings
from collections.abc import Mapping, Sequence
from contextlib import chdir
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import SparseEfficiencyWarning, csc_array, csc_matrix, csr_array, csr_matrix, issparse

from mofaflex import MOFAFLEX, likelihoods, priors, settings, terms


def compare_nested(data1, data2):
    if isinstance(data1, Mapping) and isinstance(data2, Mapping):
        if data1.keys() != data2.keys():
            return False
        return all(compare_nested(data1[k], data2[k]) for k in data1.keys())
    elif (
        isinstance(data1, Sequence | set)
        and not isinstance(data1, str | bytes)
        and isinstance(data2, Sequence | set)
        and not isinstance(data2, str | bytes)
    ):
        if len(data1) != len(data2):
            return False
        return all(compare_nested(d1, d2) for d1, d2 in zip(data1, data2, strict=False))
    elif isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        return np.all(data1 == data2)
    else:
        return data1 == data2


@pytest.fixture
def anndata_dict(random_adata, rng):
    big_adatas = (
        random_adata("Normal", 500, 100, var_names=[f"normal_var_{i}" for i in range(100)]),
        random_adata("Bernoulli", 400, 200, var_names=[f"bernoulli_var_{i}" for i in range(200)]),
        random_adata("NegativeBinomial", 600, 90, var_names=[f"negativebinomial_var_{i}" for i in range(90)]),
    )

    group_idxs = [None] * len(big_adatas)
    while True:
        for i, adata in enumerate(big_adatas):
            permuted = rng.permutation(range(adata.n_obs))
            group_size = rng.choice(np.arange(int(0.2 * adata.n_obs), int(0.8 * adata.n_obs)))
            group_idxs[i] = (permuted[:group_size], permuted[group_size:])
        intersects = [None, None]
        for i in range(2):
            intersects[i] = reduce(np.intersect1d, (idxs[i] for idxs in group_idxs))
        if all(len(intersect) > 0 for intersect in intersects):
            break

    adata_dict = {"group_1": {}, "group_2": {}}
    for view_name, (view_idx, view) in zip(
        ("view_normal", "view_bernoulli", "view_negativebinomial"), enumerate(big_adatas), strict=False
    ):
        for group_idx, group in enumerate(adata_dict.values()):
            idx = rng.choice(adata.n_vars, size=int(0.9 * adata.n_vars), replace=False)
            group[view_name] = view[group_idxs[view_idx][group_idx], idx].copy()

    adata_dict["group_1"]["view_bernoulli"].X = csr_array(adata_dict["group_1"]["view_bernoulli"].X)
    adata_dict["group_1"]["view_negativebinomial"].X = csc_array(adata_dict["group_1"]["view_negativebinomial"].X)
    adata_dict["group_2"]["view_bernoulli"].X = csr_matrix(adata_dict["group_2"]["view_bernoulli"].X)
    adata_dict["group_2"]["view_negativebinomial"].X = csc_matrix(adata_dict["group_2"]["view_negativebinomial"].X)

    return adata_dict


@pytest.fixture(scope="module")
def model_api_trained_only():
    return (
        "group_names",
        "n_groups",
        "view_names",
        "n_views",
        "feature_names",
        "n_features",
        "n_features_total",
        "sample_names",
        "n_samples",
        "n_samples_total",
        "training_loss",
        "terms",
        "likelihoods",
        "get_r2",
    )


@pytest.fixture(scope="module")
def model_api_untrained_only():
    return ("fit",)


@pytest.mark.parametrize(
    "argfor,argname,argval",
    [
        ("likelihood_normal", "scale_per_group", False),
        ("term_mofaflex", "guiding_vars_obs_keys", ["gvar_normal", "gvar_bernoulli", "gvar_categorical"]),
        ("term_mofaflex", "weight_prior", "Normal"),
        ("term_mofaflex", "weight_prior", "Laplace"),
        ("term_mofaflex", "weight_prior", "Horseshoe"),
        ("term_mofaflex", "weight_prior", priors.InformedHorseshoe(annotations_varm_key="annot_df")),
        ("term_mofaflex", "weight_prior", "SpikeSlab"),
        (
            "term_mofaflex",
            "weight_prior",
            priors.SpikeSlab(
                background_is_gaussian=True,
                init_prob=0.2,
                psi_prior_param=1.0,
                theta_prior_param_alpha=10.0,
                theta_prior_param_beta=40.0,
            ),
        ),  # GSFA settings
        ("term_mofaflex", "factor_prior", {"group_1": "Normal", "group_2": priors.Laplace()}),
        ("term_mofaflex", "factor_prior", {("group_1", "group_2"): "Laplace"}),
        ("term_mofaflex", "factor_prior", {("group_1", "group_2"): priors.Horseshoe()}),
        ("term_mofaflex", "factor_prior", "SpikeSlab"),
        (
            "term_mofaflex",
            "factor_prior",
            priors.SpikeSlab(
                background_is_gaussian=True,
                init_prob=0.2,
                psi_prior_param=1.0,
                theta_prior_param_alpha=10.0,
                theta_prior_param_beta=40.0,
            ),
        ),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_key="covar", kernel="Matern")),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_key="covar", mefisto_kernel=False)),
        ("term_mofaflex", "weight_prior", priors.GaussianProcess(covariates_key="covar", kernel="Matern")),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_mkey="covar_array", mefisto_kernel=False)),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_mkey="covar_sparse", mefisto_kernel=False)),
        (
            "term_mofaflex",
            "factor_prior",
            priors.GaussianProcess(covariates_key="covar", independent_lengthscales=True),
        ),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_key="covar", group_covar_rank=2)),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_key="covar", warp=True, warp_interval=1)),
        ("term_mofaflex", "weight_prior", priors.GaussianProcess(covariates_mkey="covar_array", mefisto_kernel=False)),
        ("term_mofaflex", "factor_prior", priors.GSFA(targets_obsm_key="perturbations_bool")),
        ("term_mofaflex", "factor_prior", priors.GSFA(targets_obsm_key="perturbations_float")),
        ("term_mofaflex", "nonnegative_weights", True),
        ("term_mofaflex", "nonnegative_factors", True),
        ("term_mofaflex", "init_factors", "orthogonal"),
        ("term_mofaflex", "init_factors", "pca"),
        ("fit", "use_obs", "intersection"),
        ("fit", "use_var", "intersection"),
        ("fit", "remove_constant_features", False),
        ("fit", "save_path", Path("test.h5")),
        ("fit", "save_path", "test.h5"),
        ("fit", "save_path", False),
    ],
)
@pytest.mark.parametrize("n_particles", [1, 5])
@pytest.mark.parametrize("batch_size", [0, 257])
@pytest.mark.parametrize("usedask", [False, True])
def test_integration(
    anndata_dict,
    tmp_path,
    argfor,
    argname,
    argval,
    n_particles,
    batch_size,
    usedask,
    model_api_trained_only,
    model_api_untrained_only,
):
    likelihoods_arg = None
    if argfor == "likelihood_normal":
        likelihoods_arg = {
            "view_normal": likelihoods.Normal(**{argname: argval}),
            "view_negativebinomial": "NegativeBinomial",
            "view_bernoulli": likelihoods.Bernoulli(),
        }

    termargs = {}
    if argfor == "term_mofaflex":
        termargs[argname] = argval
    model = terms.MofaFlex(
        n_factors=5,
        guiding_vars_likelihoods={
            "gvar_normal": "Normal",
            "gvar_bernoulli": "Bernoulli",
            "gvar_categorical": "Categorical",
        },
        **termargs,
    )
    for api in model_api_trained_only:
        with pytest.raises(RuntimeError, match="not yet trained"):
            getattr(model, api)()

    fitargs = {}
    if argfor == "fit":
        fitargs[argname] = argval
    with chdir(tmp_path), settings.override(use_dask=usedask):
        model.fit(
            anndata_dict,
            likelihoods=likelihoods_arg,
            plot_data_overview=False,
            max_epochs=2,
            seed=42,
            batch_size=batch_size,
            n_particles=n_particles,
            **fitargs,
        )

    for api in model_api_untrained_only:
        with pytest.raises(RuntimeError, match="already trained"):
            getattr(model, api)()
    for api in model_api_trained_only:
        attr = getattr(model, api)
        if callable(attr):
            attr()

    if argname == "weight_prior" and isinstance(argval, priors.InformedHorseshoe):
        assert model.n_informed_factors > 0
        assert model.terms["_"].n_informed_factors > 0
        assert model.n_informed_factors == model.terms["_"].n_informed_factors
        assert "get_annotations" in dir(model)
        assert all(isinstance(annot, pd.DataFrame) for annot in model.get_annotations().values())
    elif argname == "guiding_vars_obs_keys":
        assert model.n_guided_factors == model.terms["_"].n_guided_factors == 3
    else:
        assert (
            model.n_factors
            == model.n_total_factors
            == model.terms["_"].n_factors
            == model.terms["_"].n_total_factors
            == 5
        )
        assert "get_annotations" not in dir(model)
        with pytest.raises(AttributeError, match="is only available when using the 'InformedHorseshoe' prior"):
            model.get_annotations()

    if fitargs.get("save_path") is not False:
        loaded_model = MOFAFLEX.load(path=next(iter(tmp_path.glob("*.h5"))))

        for attr in (
            "group_names",
            "n_groups",
            "view_names",
            "n_views",
            "feature_names",
            "n_features",
            "sample_names",
            "n_samples",
            "n_samples_total",
            "n_factors",
            "n_total_factors",
            "factor_order",
            "factor_names",
        ):
            assert compare_nested(getattr(model, attr), getattr(loaded_model, attr)), attr


@pytest.mark.parametrize("usedask", [False, True])
def test_integration_single_obs(anndata_dict, usedask):
    intersection = reduce(lambda x, y: x.intersection(y), (view.obs_names for view in anndata_dict["group_2"].values()))
    anndata_dict["group_2"]["view_bernoulli"] = anndata_dict["group_2"]["view_bernoulli"][intersection[0]]
    with settings.override(use_dask=usedask):
        model = terms.MofaFlex(n_factors=5, factor_prior=priors.SpikeSlab(), weight_prior="SpikeSlab")
        model.fit(
            anndata_dict, plot_data_overview=False, use_obs="intersection", max_epochs=2, seed=42, save_path=False
        )


@pytest.mark.parametrize("usedask", [False, True])
def test_integration_single_var(anndata_dict, usedask):
    intersection = reduce(
        lambda x, y: x.intersection(y), (group["view_bernoulli"].var_names for group in anndata_dict.values())
    )
    anndata_dict["group_2"]["view_bernoulli"] = anndata_dict["group_2"]["view_bernoulli"][:, intersection[0]]
    with settings.override(use_dask=usedask):
        model = terms.MofaFlex(
            n_factors=5,
            factor_prior="SpikeSlab",
            weight_prior={("view_normal", "view_bernoulli", "view_negativebinomial"): priors.SpikeSlab()},
        )
        model.fit(
            anndata_dict, plot_data_overview=False, use_var="intersection", max_epochs=2, seed=42, save_path=False
        )


@pytest.mark.parametrize("n_particles", [1, 5])
@pytest.mark.parametrize("batch_size", [0, 257])
def test_integration_constantprior(anndata_dict, tmp_path, n_particles, batch_size):
    model = terms.MofaFlex(n_factors=5)
    model.fit(
        anndata_dict,
        plot_data_overview=False,
        max_epochs=2,
        seed=42,
        batch_size=batch_size,
        n_particles=n_particles,
        save_path=False,
    )

    factors = model.get_factors()
    weights = model.get_weights()
    with chdir(tmp_path):
        model2 = terms.MofaFlex(
            n_factors=5, factor_prior=priors.Constant(factors), weight_prior=priors.Constant(weights)
        )
        model2.fit(
            anndata_dict,
            plot_data_overview=False,
            max_epochs=2,
            seed=42,
            batch_size=batch_size,
            n_particles=n_particles,
        )
    factors2 = model2.get_factors()
    weights2 = model2.get_weights()
    for group_name, group_factors in factors.items():
        assert np.all(group_factors == factors2[group_name])
    for view_name, view_weights in weights.items():
        assert np.all(view_weights == weights2[view_name])


@pytest.mark.parametrize("n_particles", [1, 5])
@pytest.mark.parametrize("batch_size", [0, 257])
def test_integration_dynamicapi_multiple_priors(anndata_dict, tmp_path, n_particles, batch_size):
    model = terms.MofaFlex(
        n_factors=1,
        weight_prior={
            "view_normal": priors.InformedHorseshoe(annotations_varm_key="annot_df"),
            "view_negativebinomial": priors.Horseshoe(),
            "view_bernoulli": priors.Normal(),
        },
    )
    save_path = tmp_path / "informedhs_model.h5"
    with chdir(tmp_path):
        model.fit(
            anndata_dict,
            plot_data_overview=False,
            max_epochs=2,
            seed=42,
            batch_size=batch_size,
            n_particles=n_particles,
            save_path=save_path,
        )
    signif = model.get_significant_annotations()
    assert len(signif) == 1
    assert next(iter(signif.keys())) == "view_normal"
    assert signif["view_normal"]["factor"].cat.categories.size == 11

    # Loading without an explicit map_location must still set a usable device so PCGSE works.
    reloaded = MOFAFLEX.load(save_path)
    assert reloaded.get_significant_annotations().keys() == signif.keys()


@pytest.mark.parametrize("n_particles", [1, 5])
@pytest.mark.parametrize("batch_size", [0, 257])
def test_integration_informedhs_guidingvars(anndata_dict, tmp_path, n_particles, batch_size):
    for group in anndata_dict.values():
        group["view_normal"].varm["annot_normal"] = group["view_normal"].varm["annot_df"]
        group["view_negativebinomial"].varm["annot_negativebinomial"] = group["view_negativebinomial"].varm["annot_df"]
        group["view_bernoulli"].varm["annot_bernoulli"] = group["view_bernoulli"].varm["annot_df"]

    model = terms.MofaFlex(
        n_factors=3,
        weight_prior={
            "view_normal": priors.InformedHorseshoe(annotations_varm_key="annot_normal"),
            "view_negativebinomial": priors.InformedHorseshoe(annotations_varm_key="annot_negativebinomial"),
            "view_bernoulli": priors.InformedHorseshoe(annotations_varm_key="annot_bernoulli"),
        },
        guiding_vars_obs_keys=["gvar_categorical"],
        guiding_vars_likelihoods="Categorical",
        nonnegative_factors=True,
        nonnegative_weights=True,
    )
    with chdir(tmp_path):
        model.fit(
            anndata_dict,
            plot_data_overview=False,
            max_epochs=2,
            seed=42,
            batch_size=batch_size,
            n_particles=n_particles,
        )

    assert model.n_total_factors == 34
    assert model.factor_names.shape == np.unique(model.factor_names).shape


@pytest.mark.parametrize("n_particles", [1, 5])
@pytest.mark.parametrize("batch_size", [0, 257])
@pytest.mark.parametrize("select", ["max", "min"])
def test_integration_covariates_singlegroup(anndata_dict, tmp_path, select, n_particles, batch_size):
    selected_group = getattr(builtins, select)(
        (
            (group_name, len(reduce(lambda x, y: x.union(y), (view.obs_names for view in group.values()))))
            for group_name, group in anndata_dict.items()
        ),
        key=lambda x: x[1],
    )[0]
    for adata in anndata_dict[selected_group].values():
        adata.obs["test_covar"] = adata.obs["covar"]

    factor_prior = {group_name: "Normal" for group_name in anndata_dict.keys() if group_name != selected_group}
    factor_prior[selected_group] = priors.GaussianProcess(covariates_key="test_covar")

    model = terms.MofaFlex(n_factors=5, factor_prior=factor_prior)

    with chdir(tmp_path):
        model.fit(
            anndata_dict,
            plot_data_overview=False,
            max_epochs=2,
            seed=42,
            batch_size=batch_size,
            n_particles=n_particles,
        )

    assert list(model.factor_covariates.keys()) == [selected_group]


@pytest.mark.parametrize("usedask", [False, True])
def test_imputation(rng, anndata_dict, usedask):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SparseEfficiencyWarning)

        nanidx = {}
        for group_name, group in anndata_dict.items():
            cnanidx = {}
            for view_name, view in group.items():
                n_nans = rng.choice(int(0.05 * view.n_obs * view.n_vars))
                rowidx = rng.choice(view.n_obs, size=n_nans)
                colidx = rng.choice(view.n_vars, size=n_nans)

                view.X[rowidx, colidx] = np.nan
                cnanidx[view_name] = (rowidx, colidx)
            nanidx[group_name] = cnanidx

    with settings.override(use_dask=usedask):
        model = terms.MofaFlex(n_factors=5)
        with pytest.raises(RuntimeError, match="not yet trained"):
            model.impute_data()
        model.fit(anndata_dict, plot_data_overview=False, max_epochs=5, seed=42, save_path=False)

        imputed = model.impute_data(anndata_dict, missing_only=False)

    for group in imputed.values():
        for view in group.values():
            assert np.isnan(view.X if not issparse(view.X) else view.X.data).sum() == 0

    imputed = model.impute_data(anndata_dict, missing_only=True)
    dataset = model._make_dataset(anndata_dict)
    preprocessor = dataset.preprocessor
    for group_name, group in imputed.items():
        for view_name, view in group.items():
            assert np.isnan(view.X if not issparse(view.X) else view.X.data).sum() == 0

            orig_data = anndata_dict[group_name][view_name]
            new_X = view[orig_data.obs_names, orig_data.var_names].X
            orig_X = orig_data.X
            if issparse(orig_X):
                orig_X = orig_X.toarray()
            if issparse(new_X):
                new_X = new_X.toarray()
            nonnan = ~np.isnan(orig_X)
            orig_X = preprocessor(orig_X, slice(None), slice(None), group_name, view_name)[0]
            orig_X = model._model._likelihoods[view_name].transform_data(
                orig_X,
                group_name,
                dataset.map_local_indices_to_global(slice(None), group_name, view_name, align_to="samples"),
                dataset.map_local_indices_to_global(slice(None), group_name, view_name, align_to="features"),
            )
            assert np.allclose(orig_X[nonnan], new_X[nonnan])


def test_terms(anndata_dict):
    model = terms.MofaFlex("normal", n_factors=5, weight_prior="Normal") + terms.MofaFlex(
        "hs", n_factors=4, weight_prior="Horseshoe"
    )
    with pytest.raises(ValueError, match="unique term names"):
        model + terms.MofaFlex("normal", n_factors=5)
    with pytest.raises(TypeError, match="unsupported operand"):
        model + 1
    model.fit(anndata_dict, plot_data_overview=False, max_epochs=2, seed=42, save_path=False)
    with pytest.raises(ValueError, match="already trained"):
        model + terms.MofaFlex(n_factors=5)
    with pytest.raises(AttributeError):
        model.n_factors  # noqa: B018
    assert model.terms["normal"].n_factors == 5
    assert model.terms["hs"].n_factors == 4
    assert model.terms["normal"].n_samples == model.terms["hs"].n_samples == model.n_samples
