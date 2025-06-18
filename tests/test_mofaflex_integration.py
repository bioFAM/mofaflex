# integration tests: only testing if the code runs without errors
import warnings
from functools import reduce

import anndata as ad
import numpy as np
import pytest
from packaging.version import Version
from scipy.sparse import SparseEfficiencyWarning, csc_array, csc_matrix, csr_array, csr_matrix, issparse

from mofaflex import MOFAFLEX, DataOptions, ModelOptions, SmoothOptions, TrainingOptions, settings


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
    "attrname,attrvalue",
    [
        ("scale_per_group", False),
        ("scale_per_group", True),
        ("annotations_varm_key", None),
        ("covariates_obs_key", None),
        ("covariates_obs_key", "covar"),
        ("covariates_obsm_key", None),
        ("covariates_obsm_key", "covar"),
        ("guiding_vars_obs_keys", {"gvar1": "gvar_normal", "gvar2": "gvar_bernoulli", "gvar3": "gvar_categorical"}),
        ("use_obs", "union"),
        ("use_obs", "intersection"),
        ("use_var", "union"),
        ("use_var", "intersection"),
        ("remove_constant_features", True),
        ("remove_constant_features", False),
        ("weight_prior", "Normal"),
        ("weight_prior", "Laplace"),
        ("weight_prior", "Horseshoe"),
        ("weight_prior", "SnS"),
        ("factor_prior", "Normal"),
        ("factor_prior", "Laplace"),
        ("factor_prior", "Horseshoe"),
        ("factor_prior", "SnS"),
        ("nonnegative_weights", False),
        ("nonnegative_weights", True),
        ("nonnegative_factors", False),
        ("nonnegative_factors", True),
        ("init_factors", "random"),
        ("init_factors", "orthogonal"),
        ("init_factors", "pca"),
        ("save_path", None),
        ("save_path", "test.h5"),
        ("batch_size", 0),
        ("batch_size", 100),
    ],
)
@pytest.mark.parametrize("usedask", [False, True])
@pytest.mark.xfail(
    Version(ad.__version__) >= Version("0.12.0rc1") and Version(ad.__version__) < Version("0.12.0"),
    reason="anndata bug: https://github.com/scverse/anndata/pull/1975",
    strict=False,
)
def test_integration(anndata_dict, tmp_path, attrname, attrvalue, usedask):
    opts = (
        DataOptions(plot_data_overview=False, annotations_varm_key="annot"),
        ModelOptions(
            n_factors=5, guiding_vars_likelihoods={"gvar1": "Normal", "gvar2": "Bernoulli", "gvar3": "Categorical"}
        ),
        TrainingOptions(max_epochs=2, seed=42, save_path=False),
    )
    if attrname == "save_path" and isinstance(attrvalue, str):
        attrvalue = str(tmp_path / attrvalue)
    for opt in opts:
        if hasattr(opt, attrname):
            setattr(opt, attrname, attrvalue)

    with settings.override(use_dask=usedask):
        model = MOFAFLEX(anndata_dict, *opts)

    if attrname == "weight_prior" and attrvalue == "Horseshoe":
        assert (model.n_informed_factors > 0) | (model._n_guiding_vars > 0)
    elif attrname == "guiding_vars_obs_keys":
        assert model._n_guiding_vars == 3
    else:
        assert model.n_factors == model.n_dense_factors == 5


@pytest.mark.parametrize("usedask", [False, True])
def test_integration_single_obs(anndata_dict, usedask):
    intersection = reduce(lambda x, y: x.intersection(y), (view.obs_names for view in anndata_dict["group_2"].values()))
    anndata_dict["group_2"]["view_bernoulli"] = anndata_dict["group_2"]["view_bernoulli"][intersection[0]]
    with settings.override(use_dask=usedask):
        MOFAFLEX(
            anndata_dict,
            DataOptions(plot_data_overview=False, use_obs="intersection"),
            ModelOptions(n_factors=5, factor_prior="SnS", weight_prior="SnS"),
            TrainingOptions(max_epochs=2, seed=42, save_path=False),
        )


@pytest.mark.parametrize("usedask", [False, True])
def test_integration_single_var(anndata_dict, usedask):
    intersection = reduce(
        lambda x, y: x.intersection(y), (group["view_bernoulli"].var_names for group in anndata_dict.values())
    )
    anndata_dict["group_2"]["view_bernoulli"] = anndata_dict["group_2"]["view_bernoulli"][:, intersection[0]]
    with settings.override(use_dask=usedask):
        MOFAFLEX(
            anndata_dict,
            DataOptions(plot_data_overview=False, use_var="intersection"),
            ModelOptions(n_factors=5, factor_prior="SnS", weight_prior="SnS"),
            TrainingOptions(max_epochs=2, seed=42, save_path=False),
        )


@pytest.mark.parametrize(
    "attrname,attrvalue",
    [
        ("kernel", "Matern"),
        ("mefisto_kernel", False),
        ("independent_lengthscales", True),
        ("group_covar_rank", 2),
        ("warp_groups", ["group_1", "group_2"]),
    ],
)
@pytest.mark.parametrize("usedask", [False, True])
@pytest.mark.xfail(
    Version(ad.__version__) >= Version("0.12.0rc1") and Version(ad.__version__) < Version("0.12.0"),
    reason="anndata bug: https://github.com/scverse/anndata/pull/1975",
    strict=False,
)
def test_integration_gp(anndata_dict, attrname, attrvalue, usedask):
    opts = (
        DataOptions(covariates_obs_key="covar", plot_data_overview=False),
        ModelOptions(n_factors=5, factor_prior="GP"),
        TrainingOptions(max_epochs=2, seed=42, mofa_compat=True),
    )
    smooth_opts = SmoothOptions(n_inducing=20, warp_interval=1)
    setattr(smooth_opts, attrname, attrvalue)

    with settings.override(use_dask=usedask):
        model = MOFAFLEX(anndata_dict, *opts, smooth_opts)  # noqa F841


@pytest.mark.parametrize("usedask", [False, True])
@pytest.mark.xfail(
    Version(ad.__version__) >= Version("0.12.0rc1") and Version(ad.__version__) < Version("0.12.0"),
    reason="anndata bug: https://github.com/scverse/anndata/pull/1975",
    strict=False,
)
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
