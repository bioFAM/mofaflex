# integration tests: only testing if the code runs without errors
import warnings
from collections.abc import Mapping, Sequence
from contextlib import chdir
from functools import reduce
from pathlib import Path

import numpy as np
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

    group_idxs = []
    for adata in big_adatas:
        permuted = rng.permutation(range(adata.n_obs))
        group_size = rng.choice(np.arange(int(0.2 * adata.n_obs), int(0.8 * adata.n_obs)))
        group_idxs.append((permuted[:group_size], permuted[group_size:]))

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
        "get_r2",
        "get_dispersion",
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
        ("term_mofaflex", "factor_prior", {"group_1": "Normal", "group_2": priors.Laplace()}),
        ("term_mofaflex", "factor_prior", {("group_1", "group_2"): "Laplace"}),
        ("term_mofaflex", "factor_prior", {("group_1", "group_2"): priors.Horseshoe()}),
        ("term_mofaflex", "factor_prior", "SpikeSlab"),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_obs_key="covar", kernel="Matern")),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_obs_key="covar", mefisto_kernel=False)),
        (
            "term_mofaflex",
            "factor_prior",
            priors.GaussianProcess(covariates_obsm_key="covar_array", mefisto_kernel=False),
        ),
        (
            "term_mofaflex",
            "factor_prior",
            priors.GaussianProcess(covariates_obsm_key="covar_sparse", mefisto_kernel=False),
        ),
        (
            "term_mofaflex",
            "factor_prior",
            priors.GaussianProcess(covariates_obs_key="covar", independent_lengthscales=True),
        ),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_obs_key="covar", group_covar_rank=2)),
        ("term_mofaflex", "factor_prior", priors.GaussianProcess(covariates_obs_key="covar", warp=True)),
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
