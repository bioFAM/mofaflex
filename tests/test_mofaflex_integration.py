# integration tests: only testing if the code runs without errors
import warnings
from contextlib import chdir
from functools import reduce
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import SparseEfficiencyWarning, csc_array, csc_matrix, csr_array, csr_matrix, issparse

from mofaflex import likelihoods, priors, settings, terms


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
    ],
)
@pytest.mark.parametrize("n_particles", [1, 5])
@pytest.mark.parametrize("batch_size", [0, 257])
@pytest.mark.parametrize("usedask", [False, True])
def test_integration(anndata_dict, tmp_path, argfor, argname, argval, n_particles, batch_size, usedask, request):
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

    fitargs = {"save_path": False}
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
            del group["view_negativebinomial"]
            cnanidx = {}
            for view_name, view in group.items():
                n_nans = rng.choice(int(0.05 * view.n_obs * view.n_vars))
                rowidx = rng.choice(view.n_obs, size=n_nans)
                colidx = rng.choice(view.n_vars, size=n_nans)

                view.X[rowidx, colidx] = np.nan
                cnanidx[view_name] = (rowidx, colidx)
            nanidx[group_name] = cnanidx

    with settings.override(use_dask=usedask):
        model = MOFAFLEX(
            anndata_dict,
            DataOptions(plot_data_overview=False),
            ModelOptions(n_factors=5),
            TrainingOptions(max_epochs=2, seed=42, save_path=False),
        )

        imputed = model.impute_data(anndata_dict, missing_only=False)

    for group in imputed.values():
        for view in group.values():
            assert np.isnan(view.X if not issparse(view.X) else view.X.data).sum() == 0

    imputed = model.impute_data(anndata_dict, missing_only=True)
    preprocessor = model._mofaflexdataset(anndata_dict).preprocessor
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
            assert np.allclose(
                preprocessor(orig_X, slice(None), slice(None), group_name, view_name)[0][nonnan], new_X[nonnan]
            )
