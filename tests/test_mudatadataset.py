import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from mudata import MuData
from packaging.version import Version
from scipy import sparse

from mofaflex import settings
from mofaflex._core.datasets import MofaFlexDataset, MuDataDataset


@pytest.fixture(scope="module")
def mdata(rng):
    nobs = 500
    nvar_per_mod = 20
    ngroups = 4

    obs_names = [f"cell_{i}" for i in range(nobs)]
    adatas = {}
    for view in range(3):
        cobs_names = rng.choice(obs_names, size=int(0.8 * nobs), replace=False)
        adata = ad.AnnData(
            X=rng.poisson(0.5, size=(len(cobs_names), nvar_per_mod)),
            layers={"layer1": rng.normal(0, 1, size=(len(cobs_names), nvar_per_mod)).astype(np.float32)},
            obs=pd.DataFrame(index=cobs_names),
            var=pd.DataFrame(index=[f"mod_{view}_feature_{i}" for i in range(nvar_per_mod)]),
        )
        if view < 2:
            adata.obs["covar"] = rng.random(size=len(cobs_names))
            adata.obsm["covar_df"] = pd.DataFrame(
                rng.random(size=(len(cobs_names), 3)), columns=["a", "b", "c"], index=adata.obs_names
            )
            adata.obsm["covar_array"] = rng.random(size=(len(cobs_names), 3))
            adata.obsm["covar_sparse"] = sparse.csr_array(rng.poisson(size=(len(cobs_names), 3)))
            adata.var["covar"] = rng.random(size=adata.n_vars)
            adata.varm["annot_df"] = pd.DataFrame(
                rng.random(size=(nvar_per_mod, 10)),
                columns=[f"annot_view_{view}_{i}" for i in range(10)],
                index=adata.var_names,
            )
            adata.varm["annot_array"] = rng.random(size=(nvar_per_mod, 10))
            adata.varm["annot_sparse"] = sparse.csr_array(rng.poisson(size=(nvar_per_mod, 4)))
        adatas[f"view_{view}"] = adata

    adatas["view_0"].X = sparse.csr_array(adatas["view_0"].X)
    adatas["view_2"].X = sparse.csc_array(adatas["view_2"].X)
    adatas["view_1"].var["highly_variable"] = rng.choice((True, False), size=nvar_per_mod, p=[0.4, 0.6])

    mdata = md.MuData(adatas)

    mdata.obs["batch"] = pd.Categorical(rng.choice(ngroups, size=mdata.n_obs).astype(str))

    global_covar = rng.random(size=mdata.n_obs)
    global_covar[~mdata.obs_names.isin(adatas["view_2"].obs_names)] = np.nan
    mdata.obs["covar"] = global_covar

    global_covar = rng.random(size=mdata.n_vars)
    global_covar[~mdata.var_names.isin(adatas["view_2"].var_names)] = np.nan
    mdata.var["covar"] = global_covar

    global_covar = rng.random(size=(mdata.n_obs, 3))
    global_covar[~mdata.obs_names.isin(adatas["view_2"].obs_names)] = np.nan
    mdata.obsm["covar_df"] = pd.DataFrame(global_covar, columns=["a", "b", "c"], index=mdata.obs_names)
    mdata.obsm["covar_array"] = rng.permuted(global_covar, axis=0)

    global_covar_sparse = sparse.csr_array(rng.poisson(size=(mdata.n_obs, 3)).astype(np.float32))
    global_covar_sparse[~mdata.obs_names.isin(adatas["view_2"].obs_names)] = np.nan
    mdata.obsm["covar_sparse"] = global_covar_sparse

    global_annot = rng.random(size=(mdata.n_vars, 10))
    global_annot[~mdata.var_names.isin(adatas["view_2"].var_names)] = np.nan
    mdata.varm["annot_df"] = pd.DataFrame(
        global_annot, columns=[f"global_annot_{i}" for i in range(10)], index=mdata.var_names
    )
    mdata.varm["annot_array"] = rng.permuted(global_annot, axis=0)

    global_annot_sparse = sparse.csr_array(rng.poisson(size=(mdata.n_var, 4)).astype(np.float32))
    global_annot_sparse[~mdata.var_names.isin(adatas["view_2"].var_names)] = np.nan
    mdata.varm["annot_sparse"] = global_annot_sparse

    mdata.var["global_highly_variable"] = rng.choice((True, False), size=mdata.n_vars, p=[0.3, 0.7])

    return mdata


@pytest.fixture(scope="module", params=(None, "global_highly_variable", "highly_variable"))
def subset_var(request):
    return request.param


@pytest.fixture(scope="module", params=(None, "layer1", {"view_0": "layer1", "view_1": None, "view_2": "layer1"}))
def layer(request):
    return request.param


@pytest.fixture(scope="module")
def dataset(mdata, layer, subset_var):
    return MofaFlexDataset(mdata, group_by="batch", layer=layer, subset_var=subset_var, cast_to=np.float32)


def get_varnames(mdata, modname, subset_var):
    varnames = mdata.mod[modname].var_names
    if subset_var in mdata.var:
        varnames = varnames[mdata.var[subset_var][mdata.varmap[modname].reshape(-1) > 0]]
    elif subset_var in mdata[modname].var:
        varnames = varnames[mdata[modname].var[subset_var]]
    return varnames


def test_instance(dataset):
    assert isinstance(dataset, MuDataDataset)


def test_properties(mdata, dataset, subset_var):
    assert sorted(dataset.group_names) == sorted(mdata.obs["batch"].unique())
    assert sorted(dataset.view_names) == sorted(mdata.mod.keys())
    for group_name, sample_names in dataset.sample_names.items():
        cmdata = mdata[mdata.obs["batch"] == group_name, :]
        assert np.all(np.sort(sample_names) == cmdata.obs_names.sort_values().to_numpy())
        assert dataset.n_samples[group_name] == cmdata.n_obs

    for view_name, view_names in dataset.feature_names.items():
        cvarnames = get_varnames(mdata, view_name, subset_var)

        assert np.all(view_names == cvarnames)
        assert dataset.n_features[view_name] == cvarnames.size


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_alignment(mdata, dataset, rng, axis):
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

            mod = mdata[mdata.obs["batch"] == group_name, :][view_name]
            local_obsnames = group_samples[np.isin(group_samples, mod.obs_names)]
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

            mod = mdata[mdata.obs["batch"] == group_name, :][view_name]
            local_varnames = view_features[np.isin(view_features, mod.var_names)]
            idx = pd.Index(view_features).get_indexer(local_varnames)
            assert np.all(local_arr == np.take(global_arr, idx, axis=axis))

            idx = np.isin(view_features, local_varnames)
            assert np.all(np.isnan(np.compress(~idx, new_global_arr, axis=axis)))
            assert np.all(np.compress(idx, new_global_arr, axis=axis) == np.compress(idx, global_arr, axis=axis))


def test_index_mapping(mdata, dataset, rng):
    for group_name, group_samples in dataset.sample_names.items():
        global_idx = rng.choice(group_samples.size, size=int(0.3 * group_samples.size), replace=True)
        for view_name in dataset.view_names:
            local_idx = dataset.map_global_indices_to_local(global_idx, group_name, view_name, align_to="samples")

            local_obsnames = mdata[mdata.obs["batch"] == group_name, :][view_name].obs_names.intersection(
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

            local_varnames = mdata[mdata.obs["batch"] == group_name, :][view_name].var_names.intersection(
                view_features, sort=False
            )
            assert np.all(view_features[global_idx][local_idx >= 0] == local_varnames[local_idx[local_idx >= 0]])
            assert np.all(~np.isin(view_features[global_idx][local_idx < 0], local_varnames))

            new_global_idx = dataset.map_local_indices_to_global(
                local_idx[local_idx >= 0], group_name, view_name, align_to="features"
            )
            assert np.all(global_idx[local_idx >= 0] == new_global_idx)


def test_getitems(mdata, dataset, layer, rng):
    if layer is not None:
        if isinstance(layer, str):
            func = lambda modname: layer
        else:
            func = lambda modname: layer[modname]
        mods = {}
        for modname in mdata.mod.keys():
            adata = mdata.mod[modname]
            clayer = func(modname)
            mods[modname] = AnnData(
                X=adata.X if clayer is None else adata.layers[clayer],
                obs=adata.obs,
                var=adata.var,
                obsm=adata.obsm,
                varm=adata.varm,
            )
        new_mdata = MuData(mods, obs=mdata.obs, var=mdata.var, obsmap=mdata.obsmap, varmap=mdata.varmap)
        new_mdata.obs = mdata.obs
        new_mdata.var = mdata.var
        mdata = new_mdata

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
            cdata = mdata[sample_names, feature_names][view_name]
            assert np.all(cdata.X == view)

            cnonmissing_obs = np.nonzero(np.isin(sample_names, cdata.obs_names))[0]
            assert np.all(items["nonmissing_samples"][group_name][view_name] == cnonmissing_obs)
            assert items["nonmissing_features"][group_name][view_name] == slice(None)


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_group_view(mdata, dataset, usedask):
    def applyfun(adata, group_name, view_name, ref_sample_names, ref_feature_names):
        assert np.all(
            adata.obs_names == pd.Index(ref_sample_names).intersection(mdata[view_name].obs_names, sort=False)
        )
        assert np.all(adata.var_names == ref_feature_names)

    with settings.override(use_dask=usedask):
        dataset.apply(
            applyfun,
            group_kwargs={"ref_sample_names": dataset.sample_names},
            view_kwargs={"ref_feature_names": dataset.feature_names},
        )


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_view(mdata, dataset, usedask):
    def applyfun(adata, group_name, view_name):
        assert np.all(adata.obs_names.sort_values() == mdata[view_name].obs_names.sort_values())
        assert np.all(adata.var_names == dataset.feature_names[view_name])

    with settings.override(use_dask=usedask):
        dataset.apply(applyfun, by_group=False)


@pytest.mark.xfail(
    Version(ad.__version__) < Version("0.11.4"), reason="anndata bug: https://github.com/scverse/anndata/pull/1911"
)
@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_group(mdata, dataset, subset_var, usedask):
    varnames = np.concat(tuple(dataset.feature_names.values()))

    def applyfun(adata, group_name, view_name):
        assert np.all(adata.obs_names == dataset.sample_names[group_name])
        assert np.all(adata.var_names == varnames)

    with settings.override(use_dask=usedask):
        dataset.apply(applyfun, by_view=False)


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_to_view(mdata, dataset, usedask):
    def applyfun(adata, group_name, ref_sample_names, ref_feature_names, _view_name):
        assert np.all(
            adata.obs_names == pd.Index(ref_sample_names).intersection(mdata[_view_name].obs_names, sort=False)
        )
        assert np.all(adata.var_names == ref_feature_names)

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
def test_apply_to_group(mdata, dataset, usedask):
    def applyfun(adata, view_name, ref_sample_names, ref_feature_names):
        assert np.all(
            adata.obs_names == pd.Index(ref_sample_names).intersection(mdata[view_name].obs_names, sort=False)
        )
        assert np.all(adata.var_names == ref_feature_names)

    with settings.override(use_dask=usedask):
        for group_name in dataset.group_names:
            dataset.apply_to_group(
                group_name,
                applyfun,
                view_kwargs={"ref_feature_names": dataset.feature_names},
                ref_sample_names=dataset.sample_names[group_name],
            )


@pytest.mark.parametrize("axis", [0, 1])
def test_get_covariates_from_key(mdata, dataset, subset_var, axis):
    attr = "obs" if axis == 0 else "var"
    namesattr = f"{attr}_names"
    dsetnamesattr = "sample_names" if axis == 0 else "feature_names"
    dict_reorder = slice(None) if axis == 0 else slice(None, None, -1)

    covars = dataset.get_covariates(axis=axis, key="covar")

    for group_name in dataset.group_names:
        subdata = mdata[mdata.obs["batch"] == group_name]
        for modname in subdata.mod.keys():
            dict_key = (group_name, modname)[dict_reorder]
            assert covars[dict_key[0]][dict_key[1]].columns == ["covar"]

            names = getattr(dataset, dsetnamesattr)[dict_key[0]]
            if modname != "view_2":
                view = subdata.mod[modname][:, get_varnames(subdata, modname, subset_var)]
                covar = getattr(view, attr)["covar"].to_numpy()
                globalidx = np.isin(names, getattr(view, namesattr))
                localidx = getattr(view, namesattr).get_indexer(names)
                localidx = localidx[localidx >= 0]
                assert np.all(
                    covars[dict_key[0]][dict_key[1]].iloc[globalidx, 0].squeeze()
                    == getattr(view, attr)["covar"].to_numpy()[localidx]
                )
                assert np.all(np.isnan(covars[dict_key[0]][dict_key[1]].iloc[~globalidx, 0]))
            else:
                covar = getattr(subdata, attr).loc[names, "covar"].to_numpy()
                nanidx = np.isnan(covar)
                assert np.all(covar[~nanidx] == covars[dict_key[0]][dict_key[1]].iloc[~nanidx, 0].squeeze())
                assert np.all(np.isnan(covars[dict_key[0]][dict_key[1]].iloc[nanidx, 0]))


@pytest.mark.parametrize("type", ("df", "array", "sparse"))
@pytest.mark.parametrize("axis, mkey", [(0, "covar"), (1, "annot")])
def test_get_covariates_from_keym(mdata, dataset, subset_var, axis, mkey, type):
    mkey = f"{mkey}_{type}"
    attr = "obs" if axis == 0 else "var"
    attrm = f"{attr}m"
    namesattr = f"{attr}_names"
    dsetnamesattr = "sample_names" if axis == 0 else "feature_names"
    dict_reorder = slice(None) if axis == 0 else slice(None, None, -1)

    covars = dataset.get_covariates(axis=axis, mkey=mkey)

    for group_name in dataset.group_names:
        for modname in mdata.mod.keys():
            dict_key = (group_name, modname)[dict_reorder]
            names = getattr(dataset, dsetnamesattr)[dict_key[0]]
            subdata = mdata[dataset.sample_names[group_name], dataset.feature_names[modname]]
            ccovars = covars[dict_key[0]][dict_key[1]]
            if modname != "view_2":
                view = subdata.mod[modname]
                covar = getattr(view, attrm)[mkey]
                if type == "df":
                    assert np.all(ccovars.columns == getattr(view, attrm)[mkey].columns)
                    covar = covar.to_numpy()

                globalidx = np.isin(names, getattr(view, namesattr))
                localidx = getattr(view, namesattr).get_indexer(names)
                localidx = localidx[localidx >= 0]

                gt = getattr(view, attrm)[mkey]
                if type == "df":
                    gt = gt.iloc[localidx, :].to_numpy()
                else:
                    gt = gt[localidx, :]
                if type == "sparse":
                    gt = gt.toarray()
                assert np.all(ccovars.iloc[globalidx, :] == gt)
                assert np.all(np.isnan(ccovars.iloc[~globalidx, :]))
            else:
                covar = getattr(subdata, attrm)[mkey]
                if type == "df":
                    assert np.all(ccovars.columns == covar.columns)
                    covar = covar.to_numpy()
                elif type == "sparse":
                    covar = covar.toarray()
                ccovars = ccovars.to_numpy()
                nanidx = np.isnan(covar)
                assert np.all(covar[~nanidx] == ccovars[~nanidx])
                assert np.all(np.isnan(ccovars[nanidx]))


def test_get_missing_obs(mdata, dataset):
    missing = dataset.get_missing_obs()
    for (group_name, view_name), df in missing.groupby(["group", "view"]):
        view = mdata[mdata.obs["batch"] == group_name, :][view_name]
        cmissing = ~np.isin(dataset.sample_names[group_name], view.obs_names)
        df = df.set_index("obs_name")
        assert np.all(df.loc[dataset.sample_names[group_name], "missing"][cmissing])
        assert np.all(~df.loc[dataset.sample_names[group_name], "missing"][~cmissing])
