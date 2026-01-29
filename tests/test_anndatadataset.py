import anndata as ad
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from mofaflex import settings
from mofaflex._core.datasets import AnnDataDataset, MofaFlexDataset


@pytest.fixture(scope="module", params=[False, True])
def adata(rng, request):
    make_sparse = request.param
    nobs = 500
    nvar = 20
    ngroups = 4

    obs_names = rng.choice([f"cell_{i}" for i in range(nobs)], size=nobs, replace=False)
    var_names = [f"feature_{i}" for i in range(nvar)]
    adata = ad.AnnData(
        X=rng.poisson(0.5, size=(nobs, nvar)),
        layers={"layer1": rng.normal(0, 1, size=(nobs, nvar)).astype(np.float32)},
        obs=pd.DataFrame(
            {"covar": rng.random(size=nobs), "batch": pd.Categorical(rng.choice(ngroups, size=nobs).astype(str))},
            index=obs_names,
        ),
        var=pd.DataFrame(
            {"covar": rng.random(size=nvar), "highly_variable": rng.choice((True, False), size=nvar, p=[0.4, 0.6])},
            index=var_names,
        ),
        obsm={
            "covar_df": pd.DataFrame(rng.random(size=(nobs, 3)), columns=["a", "b", "c"], index=obs_names),
            "covar_array": rng.random(size=(nobs, 3)),
            "covar_sparse": sparse.csr_array(rng.poisson(size=(nobs, 3))),
        },
        varm={
            "annot_df": pd.DataFrame(
                rng.random(size=(nvar, 10)), columns=[f"annot_{i}" for i in range(10)], index=var_names
            ),
            "annot_array": rng.random(size=(nvar, 10)),
            "annot_sparse": sparse.csr_array(rng.poisson(size=(nvar, 4))),
        },
    )
    if make_sparse:
        adata.X = sparse.csr_array(adata.X)

    return adata


@pytest.fixture(scope="module", params=(None, "highly_variable"))
def subset_var(request):
    return request.param


@pytest.fixture(scope="module", params=(None, "layer1"))
def layer(request):
    return request.param


@pytest.fixture(scope="module")
def dataset(adata, layer, subset_var):
    return MofaFlexDataset(adata, group_by="batch", layer=layer, subset_var=subset_var, cast_to=np.float32)


def get_varnames(adata, subset_var):
    varnames = adata.var_names
    if subset_var in adata.var:
        varnames = varnames[adata.var[subset_var]]
    return varnames


def test_instance(dataset):
    assert isinstance(dataset, AnnDataDataset)


def test_properties(adata, dataset, subset_var):
    assert sorted(dataset.group_names) == sorted(adata.obs["batch"].unique())
    assert len(dataset.view_names) == 1
    for group_name, sample_names in dataset.sample_names.items():
        cadata = adata[adata.obs["batch"] == group_name, :]
        assert np.all(np.sort(sample_names) == cadata.obs_names.sort_values().to_numpy())
        assert dataset.n_samples[group_name] == cadata.n_obs

    for view_name, view_names in dataset.feature_names.items():
        cvarnames = get_varnames(adata, subset_var)

        assert np.all(view_names == cvarnames)
        assert dataset.n_features[view_name] == cvarnames.size


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_alignment(adata, dataset, rng, axis):
    ndim = 3

    arr_shape = np.asarray([2] * ndim)

    for group_name, group_samples in dataset.sample_names.items():
        arr_shape[axis] = group_samples.size
        global_arr = rng.random(size=arr_shape)
        for view_name in dataset.view_names:
            local_arr = dataset.align_global_array_to_local(
                global_arr, group_name, view_name, align_to="samples", axis=axis
            )
            new_global_arr = dataset.align_local_array_to_global(
                local_arr, group_name, view_name, align_to="samples", axis=axis, fill_value=np.nan
            )
            new_local_arr = dataset.align_global_array_to_local(
                new_global_arr, group_name, view_name, align_to="samples", axis=axis
            )

            assert new_global_arr.shape == global_arr.shape
            assert new_local_arr.shape == local_arr.shape
            assert np.all(new_local_arr == local_arr)

            subdata = adata[adata.obs["batch"] == group_name, :]
            local_obsnames = group_samples[np.isin(group_samples, subdata.obs_names)]
            idx = pd.Index(group_samples).get_indexer(local_obsnames)
            assert np.all(local_arr == np.take(global_arr, idx, axis=axis))

            idx = np.isin(group_samples, local_obsnames)
            assert np.all(np.isnan(np.compress(~idx, new_global_arr, axis=axis)))
            assert np.all(np.compress(idx, new_global_arr, axis=axis) == np.compress(idx, global_arr, axis=axis))

    for view_name, view_features in dataset.feature_names.items():
        arr_shape[axis] = view_features.size
        global_arr = rng.random(size=arr_shape)
        for group_name in dataset.group_names:
            local_arr = dataset.align_global_array_to_local(
                global_arr, group_name, view_name, align_to="features", axis=axis
            )
            new_global_arr = dataset.align_local_array_to_global(
                local_arr, group_name, view_name, align_to="features", axis=axis, fill_value=np.nan
            )
            new_local_arr = dataset.align_global_array_to_local(
                new_global_arr, group_name, view_name, align_to="features", axis=axis
            )

            assert new_global_arr.shape == global_arr.shape
            assert new_local_arr.shape == local_arr.shape
            assert np.all(new_local_arr == local_arr)

            subdata = adata[adata.obs["batch"] == group_name, :]
            local_varnames = view_features[np.isin(view_features, subdata.var_names)]
            idx = pd.Index(view_features).get_indexer(local_varnames)
            assert np.all(local_arr == np.take(global_arr, idx, axis=axis))

            idx = np.isin(view_features, local_varnames)
            assert np.all(np.isnan(np.compress(~idx, new_global_arr, axis=axis)))
            assert np.all(np.compress(idx, new_global_arr, axis=axis) == np.compress(idx, global_arr, axis=axis))


def test_index_mapping(adata, dataset, rng):
    for group_name, group_samples in dataset.sample_names.items():
        global_idx = rng.choice(group_samples.size, size=int(0.3 * group_samples.size), replace=True)
        for view_name in dataset.view_names:
            local_idx = dataset.map_global_indices_to_local(global_idx, group_name, view_name, align_to="samples")

            local_obsnames = adata[adata.obs["batch"] == group_name, :].obs_names.intersection(
                group_samples, sort=False
            )
            assert np.all(group_samples[global_idx][local_idx >= 0] == local_obsnames[local_idx[local_idx >= 0]])
            assert np.all(~np.isin(group_samples[global_idx][local_idx < 0], local_obsnames))

            new_global_idx = dataset.map_local_indices_to_global(
                local_idx[local_idx >= 0], group_name, view_name, align_to="samples"
            )
            assert np.all(global_idx[local_idx >= 0] == new_global_idx)

    for view_name, view_features in dataset.feature_names.items():
        global_idx = rng.choice(view_features.size, size=int(0.3 * view_features.size), replace=True)
        for group_name in dataset.group_names:
            local_idx = dataset.map_global_indices_to_local(global_idx, group_name, view_name, align_to="features")

            local_varnames = adata[adata.obs["batch"] == group_name, :].var_names.intersection(
                view_features, sort=False
            )
            assert np.all(view_features[global_idx][local_idx >= 0] == local_varnames[local_idx[local_idx >= 0]])
            assert np.all(~np.isin(view_features[global_idx][local_idx < 0], local_varnames))

            new_global_idx = dataset.map_local_indices_to_global(
                local_idx[local_idx >= 0], group_name, view_name, align_to="features"
            )
            assert np.all(global_idx[local_idx >= 0] == new_global_idx)


def test_getitems(adata, dataset, layer, rng):
    if layer is not None:
        adata = AnnData(adata.layers[layer], obs=adata.obs, var=adata.var)

    idx = {
        group_name: rng.choice(sample_names.size, size=sample_names.size // 3, replace=False)
        for group_name, sample_names in dataset.sample_names.items()
    }

    items = dataset.__getitems__(idx)
    for group_name, group in items["data"].items():
        sample_names = dataset.sample_names[group_name][idx[group_name]]
        assert np.all(items["sample_idx"][group_name] == idx[group_name])
        for view_name, view in group.items():
            assert type(view) is np.ndarray
            assert view.dtype == np.float32

            feature_names = dataset.feature_names[view_name]
            cdata = adata[sample_names, feature_names]
            assert np.all(cdata.X == view)

            assert np.all(items["nonmissing_samples"][group_name][view_name] == slice(None))
            assert items["nonmissing_features"][group_name][view_name] == slice(None)


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_group_view(adata, dataset, usedask):
    def applyfun(cadata, group_name, view_name, ref_sample_names, ref_feature_names):
        assert np.all(cadata.obs_names == pd.Index(ref_sample_names).intersection(adata.obs_names, sort=False))
        assert np.all(cadata.var_names == ref_feature_names)

    with settings.override(use_dask=usedask):
        dataset.apply(
            applyfun,
            group_kwargs={"ref_sample_names": dataset.sample_names},
            view_kwargs={"ref_feature_names": dataset.feature_names},
        )


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_view(adata, dataset, usedask):
    def applyfun(cadata, group_name, view_name):
        assert np.all(cadata.obs_names.sort_values() == adata.obs_names.sort_values())
        assert np.all(cadata.var_names == dataset.feature_names[view_name])

    with settings.override(use_dask=usedask):
        dataset.apply(applyfun, by_group=False)


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_group(adata, dataset, subset_var, usedask):
    varnames = np.concat(tuple(dataset.feature_names.values()))

    def applyfun(cadata, group_name, view_name):
        assert np.all(cadata.obs_names == dataset.sample_names[group_name])
        assert np.all(cadata.var_names == varnames)

    with settings.override(use_dask=usedask):
        dataset.apply(applyfun, by_view=False)


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_to_view(adata, dataset, usedask):
    def applyfun(cadata, group_name, ref_sample_names, ref_feature_names, _view_name):
        assert np.all(cadata.obs_names == pd.Index(ref_sample_names).intersection(adata.obs_names, sort=False))
        assert np.all(cadata.var_names == ref_feature_names)

    with settings.override(use_dask=usedask):
        for view_name in dataset.view_names:
            dataset.apply_to_view(
                view_name,
                applyfun,
                group_kwargs={"ref_sample_names": dataset.sample_names},
                ref_feature_names=dataset.feature_names[view_name],
                _view_name=view_name,
            )


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_to_group(adata, dataset, usedask):
    def applyfun(cadata, view_name, ref_sample_names, ref_feature_names):
        assert np.all(cadata.obs_names == pd.Index(ref_sample_names).intersection(adata.obs_names, sort=False))
        assert np.all(cadata.var_names == ref_feature_names)

    with settings.override(use_dask=usedask):
        for group_name in dataset.group_names:
            dataset.apply_to_group(
                group_name,
                applyfun,
                view_kwargs={"ref_feature_names": dataset.feature_names},
                ref_sample_names=dataset.sample_names[group_name],
            )


@pytest.mark.parametrize("axis", [0, 1])
def test_get_covariates_from_key(adata, dataset, subset_var, axis):
    attr = "obs" if axis == 0 else "var"
    namesattr = f"{attr}_names"
    dsetnamesattr = "sample_names" if axis == 0 else "feature_names"
    dict_reorder = slice(None) if axis == 0 else slice(None, None, -1)

    covars = dataset.get_covariates(axis=axis, key="covar")

    view_name = next(iter(dataset.view_names))
    for group_name in dataset.group_names:
        subdata = adata[adata.obs["batch"] == group_name]
        dict_key = (group_name, view_name)[dict_reorder]
        assert covars[dict_key[0]][dict_key[1]].columns == ["covar"]

        names = getattr(dataset, dsetnamesattr)[dict_key[0]]
        view = subdata[:, get_varnames(subdata, subset_var)]
        globalidx = np.isin(names, getattr(view, namesattr))
        localidx = getattr(view, namesattr).get_indexer(names)
        localidx = localidx[localidx >= 0]
        assert np.all(
            covars[dict_key[0]][dict_key[1]].iloc[globalidx, 0].squeeze()
            == getattr(view, attr)["covar"].to_numpy()[localidx]
        )
        assert np.all(np.isnan(covars[dict_key[0]][dict_key[1]].iloc[~globalidx, 0]))


@pytest.mark.parametrize("type", ("df", "array", "sparse"))
@pytest.mark.parametrize("axis, mkey", [(0, "covar"), (1, "annot")])
def test_get_covariates_from_keym(adata, dataset, subset_var, axis, mkey, type):
    mkey = f"{mkey}_{type}"
    attr = "obs" if axis == 0 else "var"
    attrm = f"{attr}m"
    namesattr = f"{attr}_names"
    dsetnamesattr = "sample_names" if axis == 0 else "feature_names"
    dict_reorder = slice(None) if axis == 0 else slice(None, None, -1)

    covars = dataset.get_covariates(axis=axis, mkey=mkey)

    view_name = next(iter(dataset.view_names))
    for group_name in dataset.group_names:
        dict_key = (group_name, view_name)[dict_reorder]
        names = getattr(dataset, dsetnamesattr)[dict_key[0]]
        subdata = adata[dataset.sample_names[group_name], dataset.feature_names[view_name]]
        ccovars = covars[dict_key[0]][dict_key[1]]

        globalidx = np.isin(names, getattr(subdata, namesattr))
        localidx = getattr(subdata, namesattr).get_indexer(names)
        localidx = localidx[localidx >= 0]

        gt = getattr(subdata, attrm)[mkey]
        if type == "df":
            gt = gt.iloc[localidx, :].to_numpy()
        else:
            gt = gt[localidx, :]
        if type == "sparse":
            gt = gt.toarray()
        assert np.all(ccovars.iloc[globalidx, :] == gt)
        assert np.all(np.isnan(ccovars.iloc[~globalidx, :]))


def test_get_missing_obs(adata, dataset):
    missing = dataset.get_missing_obs()
    for group_name, df in missing.groupby("group"):
        subdata = adata[adata.obs["batch"] == group_name, :]
        cmissing = ~np.isin(dataset.sample_names[group_name], subdata.obs_names)
        df = df.set_index("obs_name")
        assert np.all(df.loc[dataset.sample_names[group_name], "missing"][cmissing])
        assert np.all(~df.loc[dataset.sample_names[group_name], "missing"][~cmissing])
