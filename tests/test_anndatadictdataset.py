from functools import reduce

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from mofaflex import settings
from mofaflex._core.datasets import AnnDataDictDataset, MofaFlexDataset


@pytest.fixture(scope="module")
def anndata_dict(random_adata, rng):
    big_adata = random_adata("Normal", 500, 100)
    permuted = rng.permutation(range(500))
    group1_size = rng.choice(500)
    group_idxs = (permuted[:group1_size], permuted[group1_size:])

    permuted = rng.permutation(range(100))
    view1_size = rng.choice(100)
    view_idxs = permuted[:view1_size], permuted[view1_size:]

    adata_dict = {}

    for group_name, group_idx in enumerate(group_idxs):
        group = {}
        for view_name, view_idx in enumerate(view_idxs):
            cgroup_idx = rng.choice(group_idx, size=int(0.8 * group_idx.size), replace=False)
            cview_idx = rng.choice(view_idx, size=int(0.8 * view_idx.size), replace=False)

            group[f"view_{view_name}"] = big_adata[cgroup_idx, cview_idx].copy()
        adata_dict[f"group_{group_name}"] = group

    adata_dict["group_0"]["view_0"].X = sparse.csr_array(adata_dict["group_0"]["view_0"].X)
    adata_dict["group_1"]["view_1"].X = sparse.csc_array(adata_dict["group_1"]["view_1"].X)

    variable_genes = rng.choice(big_adata.var_names[view_idxs[0]], size=int(0.4 * view_idxs[0].size), replace=False)
    for group in adata_dict.values():
        view = group["view_0"]
        view.var["highly_variable"] = False
        view.var.loc[view.var_names.intersection(variable_genes), "highly_variable"] = True

    return adata_dict


@pytest.fixture(scope="module", params=("union", "intersection"))
def use_obs(request):
    return request.param


@pytest.fixture(scope="module", params=("union", "intersection"))
def use_var(request):
    return request.param


@pytest.fixture(scope="module", params=(None, "highly_variable"))
def subset_var(request):
    return request.param


@pytest.fixture(
    scope="module",
    params=(
        None,
        "layer1",
        {"view_0": "layer1", "view_1": None},
        {"group_0": {"view_0": None, "view_1": "layer1"}, "group_1": {"view_0": "layer1", "view_1": "layer1"}},
    ),
)
def layer(request):
    return request.param


@pytest.fixture(scope="module")
def dataset(anndata_dict, layer, use_obs, use_var, subset_var):
    return MofaFlexDataset(
        anndata_dict, layer=layer, use_obs=use_obs, use_var=use_var, subset_var=subset_var, cast_to=np.float32
    )


def test_dataset(dataset):
    assert isinstance(dataset, AnnDataDictDataset)


def test_properties(anndata_dict, use_obs, use_var, subset_var, dataset):
    obs_func = (lambda x, y: x.union(y)) if use_obs == "union" else lambda x, y: x.intersection(y)
    var_func = (lambda x, y: x.union(y)) if use_var == "union" else lambda x, y: x.intersection(y)

    obs_names = {
        group_name: reduce(obs_func, (view.obs_names for view in group.values()))
        for group_name, group in anndata_dict.items()
    }

    var_names = {}
    view_names = set()
    for group in anndata_dict.values():
        for view_name, view in group.items():
            view_names.add(view_name)
            cvarnames = view.var_names
            if subset_var is not None and subset_var in view.var.columns:
                cvarnames = cvarnames[view.var[subset_var]]

            if view_name not in var_names:
                var_names[view_name] = cvarnames
            else:
                var_names[view_name] = var_func(var_names[view_name], cvarnames)

    assert sorted(anndata_dict.keys()) == sorted(dataset.group_names)
    assert sorted(view_names) == sorted(dataset.view_names)
    for group_name, group_obs in obs_names.items():
        assert dataset.n_samples[group_name] == group_obs.size
        assert np.all(np.sort(dataset.sample_names[group_name]) == group_obs.sort_values().to_numpy())

    for view_name, view_vars in var_names.items():
        assert dataset.n_features[view_name] == view_vars.size
        assert np.all(np.sort(dataset.feature_names[view_name]) == view_vars.sort_values().to_numpy())


@pytest.mark.parametrize("axis", (0, 1, 2))
def test_alignment(anndata_dict, dataset, rng, axis):
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

            local_obsnames = anndata_dict[group_name][view_name].obs_names.intersection(group_samples, sort=False)
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
                new_global_arr, group_name, view_name, align_to="featurs", axis=axis
            )

            assert new_local_arr.shape == local_arr.shape
            assert np.all(new_local_arr == local_arr)
            assert np.isnan(new_global_arr).sum() == global_arr.size - local_arr.size

            local_varnames = anndata_dict[group_name][view_name].var_names.intersection(view_features, sort=False)
            idx = pd.Index(view_features).get_indexer(local_varnames)
            assert np.all(local_arr == np.take(global_arr, idx, axis=axis))

            idx = np.isin(view_features, local_varnames)
            assert np.all(np.isnan(np.compress(~idx, new_global_arr, axis=axis)))
            assert np.all(np.compress(idx, new_global_arr, axis=axis) == np.compress(idx, global_arr, axis=axis))


def test_index_mapping(anndata_dict, dataset, rng):
    for group_name, group_samples in dataset.sample_names.items():
        global_idx = rng.choice(group_samples.size, size=int(0.3 * group_samples.size), replace=True)
        for view_name in dataset.view_names:
            local_idx = dataset.map_global_indices_to_local(global_idx, group_name, view_name, align_to="samples")

            local_obsnames = anndata_dict[group_name][view_name].obs_names.intersection(group_samples, sort=False)
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

            local_varnames = anndata_dict[group_name][view_name].var_names.intersection(view_features, sort=False)
            assert np.all(view_features[global_idx][local_idx >= 0] == local_varnames[local_idx[local_idx >= 0]])
            assert np.all(~np.isin(view_features[global_idx][local_idx < 0], local_varnames))

            new_global_idx = dataset.map_local_indices_to_global(
                local_idx[local_idx >= 0], group_name, view_name, align_to="features"
            )
            assert np.all(global_idx[local_idx >= 0] == new_global_idx)


def test_getitems(anndata_dict, dataset, layer, rng):
    get_layer = lambda adata, layer: adata.layers[layer] if layer is not None else adata.X
    if layer is not None:
        if isinstance(layer, str):
            func = lambda group_name, view_name: layer
        elif isinstance(layer, dict) and all(isinstance(view, str | None) for view in layer.values()):
            func = lambda group_name, view_name: layer[view_name]
        else:
            func = lambda group_name, view_name: layer[group_name][view_name]
        anndata_dict = {
            group_name: {
                view_name: AnnData(
                    X=get_layer(view, func(group_name, view_name)),
                    obs=view.obs,
                    var=view.var,
                    obsm=view.obsm,
                    varm=view.varm,
                )
                for view_name, view in group.items()
            }
            for group_name, group in anndata_dict.items()
        }

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
            cadata = anndata_dict[group_name][view_name]

            cobsidx = np.isin(sample_names, cadata.obs_names)
            cvaridx = np.isin(feature_names, cadata.var_names)

            cobs = sample_names[cobsidx]
            cvar = feature_names[cvaridx]
            assert np.all(cadata[cobs, cvar].X == view)

            cnonmissing_obs = np.nonzero(cobsidx)[0]
            cnonmissing_var = np.nonzero(cvaridx)[0]
            assert np.all(items["nonmissing_samples"][group_name][view_name] == cnonmissing_obs)
            assert np.all(items["nonmissing_features"][group_name][view_name] == cnonmissing_var)


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_group_view(anndata_dict, dataset, usedask):
    def applyfun(adata, group_name, view_name, ref_adata, ref_sample_names, ref_feature_names):
        assert np.all(adata.obs_names == ref_adata.obs_names.intersection(ref_sample_names))
        assert np.all(adata.var_names == ref_adata.var_names.intersection(ref_feature_names))

    with settings.override(use_dask=usedask):
        dataset.apply(
            applyfun,
            group_kwargs={"ref_sample_names": dataset.sample_names},
            view_kwargs={"ref_feature_names": dataset.feature_names},
            group_view_kwargs={"ref_adata": anndata_dict},
        )


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_view(anndata_dict, dataset, usedask):
    def applyfun(adata, group_name, view_name):
        view_obs = reduce(lambda x, y: x.union(y), (group[view_name].obs_names for group in anndata_dict.values()))
        view_obs = view_obs.intersection(np.concatenate(list(dataset.sample_names.values())))

        assert np.all(np.sort(adata.obs_names) == view_obs.sort_values())
        assert np.all(adata.var_names == dataset.feature_names[view_name])

    with settings.override(use_dask=usedask):
        dataset.apply(applyfun, by_group=False)


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_by_group(anndata_dict, dataset, usedask):
    def applyfun(adata, group_name, view_name):
        group_var = reduce(lambda x, y: x.union(y), (view.var_names for view in anndata_dict[group_name].values()))
        group_var = group_var.intersection(np.concatenate(list(dataset.feature_names.values())))

        assert np.all(adata.obs_names == dataset.sample_names[group_name])
        assert np.all(np.sort(adata.var_names) == group_var.sort_values())

    with settings.override(use_dask=usedask):
        dataset.apply(applyfun, by_view=False)


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_to_view(anndata_dict, dataset, usedask):
    def applyfun(adata, group_name, ref_adata, ref_sample_names, ref_feature_names, _view_name):
        assert np.all(adata.obs_names == ref_adata[group_name][_view_name].obs_names.intersection(ref_sample_names))
        assert np.all(adata.var_names == ref_adata[group_name][_view_name].var_names.intersection(ref_feature_names))

    with settings.override(use_dask=usedask):
        for view_name in dataset.view_names:
            dataset.apply_to_view(
                view_name,
                applyfun,
                group_kwargs={"ref_sample_names": dataset.sample_names},
                ref_adata=anndata_dict,
                ref_feature_names=dataset.feature_names[view_name],
                _view_name=view_name,
            )


@pytest.mark.parametrize("usedask", [False, True])
def test_apply_to_group(anndata_dict, dataset, usedask):
    def applyfun(adata, view_name, ref_adata, ref_sample_names, ref_feature_names):
        assert np.all(adata.obs_names == ref_adata[view_name].obs_names.intersection(ref_sample_names))
        assert np.all(adata.var_names == ref_adata[view_name].var_names.intersection(ref_feature_names))

    with settings.override(use_dask=usedask):
        for group_name in dataset.group_names:
            dataset.apply_to_group(
                group_name,
                applyfun,
                view_kwargs={"ref_feature_names": dataset.feature_names},
                ref_adata=anndata_dict[group_name],
                ref_sample_names=dataset.sample_names[group_name],
            )


@pytest.mark.parametrize("axis", [0, 1])
def test_get_covariates_from_key(anndata_dict, dataset, axis):
    attr = "obs" if axis == 0 else "var"
    namesattr = f"{attr}_names"
    dsetnamesattr = "sample_names" if axis == 0 else "feature_names"
    dict_reorder = slice(None) if axis == 0 else slice(None, None, -1)

    covars = dataset.get_covariates(axis=axis, key="covar")

    for group_name, group in anndata_dict.items():
        for view_name, view in group.items():
            dict_key = (group_name, view_name)[dict_reorder]
            assert covars[dict_key[0]][dict_key[1]].columns == ["covar"]

            names = getattr(dataset, dsetnamesattr)[dict_key[0]]
            globalidx = np.isin(names, getattr(view, namesattr))
            localidx = getattr(view, namesattr).get_indexer(names)
            localidx = localidx[localidx >= 0]

            assert np.all(
                covars[dict_key[0]][dict_key[1]].iloc[globalidx].squeeze()
                == getattr(view, attr)["covar"].to_numpy()[localidx]
            )
            assert np.all(np.isnan(covars[dict_key[0]][dict_key[1]][~globalidx]))


@pytest.mark.parametrize("type", ("df", "array", "sparse"))
@pytest.mark.parametrize("axis, mkey", [(0, "covar"), (1, "annot")])
def test_get_covariates_from_keym(anndata_dict, dataset, axis, mkey, type):
    mkey = f"{mkey}_{type}"
    attr = "obs" if axis == 0 else "var"
    attrm = f"{attr}m"
    namesattr = f"{attr}_names"
    dsetnamesattr = "sample_names" if axis == 0 else "feature_names"
    dict_reorder = slice(None) if axis == 0 else slice(None, None, -1)

    covars = dataset.get_covariates(axis=axis, mkey=mkey)

    for group_name, group in anndata_dict.items():
        for view_name, view in group.items():
            dict_key = (group_name, view_name)[dict_reorder]
            if type == "df":
                assert np.all(covars[dict_key[0]][dict_key[1]].columns == getattr(view, attrm)[mkey].columns)

            names = getattr(dataset, dsetnamesattr)[dict_key[0]]
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
            ccovars = covars[dict_key[0]][dict_key[1]].to_numpy()
            assert np.all(ccovars[globalidx, :] == gt)
            assert np.all(pd.isnull(ccovars[~globalidx, :]))


def test_get_missing_obs(anndata_dict, dataset):
    missing = dataset.get_missing_obs()
    for (group_name, view_name), df in missing.groupby(["group", "view"]):
        view = anndata_dict[group_name][view_name]
        cmissing = ~np.isin(dataset.sample_names[group_name], view.obs_names)
        df = df.set_index("obs_name")
        assert np.all(df.loc[dataset.sample_names[group_name], "missing"][cmissing])
        assert np.all(~df.loc[dataset.sample_names[group_name], "missing"][~cmissing])
