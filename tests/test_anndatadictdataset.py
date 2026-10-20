import operator
from collections import defaultdict
from collections.abc import Mapping
from functools import reduce

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

from mofaflex import settings
from mofaflex._core.datasets import AnnDataDictDataset, MofaFlexDataset

_LAYERSPECS_SAME_FOR_ALL = (None, "layer1", {"x1": None, "x2": "layer1"}, {"x1": "layer1", "x2": "layer2"})
_LAYERSPECS_SEPARATE_PER_VIEW = reduce(
    operator.add, ([{"view_0": spec}, {"view_0": spec, "view_1": None}] for spec in _LAYERSPECS_SAME_FOR_ALL)
)
_LAYERSPECS_SEPARATE_PER_GROUP_VIEW = reduce(
    operator.add,
    (
        [
            {"group_0": {"view_0": spec}, "group_1": {"view_0": spec}},
            {"group_0": {"view_0": spec, "view_1": None}, "group_1": {"view_0": spec, "view_1": None}},
        ]
        for spec in _LAYERSPECS_SAME_FOR_ALL
    ),
)


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
    params=(*_LAYERSPECS_SAME_FOR_ALL, *_LAYERSPECS_SEPARATE_PER_VIEW, *_LAYERSPECS_SEPARATE_PER_GROUP_VIEW),
)
def layer(request):
    return request.param


@pytest.fixture(scope="function")
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


@pytest.mark.parametrize("layerspec", _LAYERSPECS_SAME_FOR_ALL)
def test_parse_layer_spec_same_for_all(anndata_dict, layerspec):
    layerfunc = AnnDataDictDataset._parse_layer_spec(layerspec, anndata_dict)

    view_names = reduce(operator.or_, (group.keys() for group in anndata_dict.values()))
    layers = [layerfunc(group_name, view_name) for group_name in anndata_dict.keys() for view_name in view_names]
    assert all(layers[0] == layer for layer in layers[1:])
    layer = layers[0]
    if not isinstance(layerspec, Mapping):
        assert layer == {None: layerspec}
    else:
        assert layer == layerspec


@pytest.mark.parametrize("layerspec", _LAYERSPECS_SEPARATE_PER_VIEW)
def test_parse_layer_spec_separate_per_view(anndata_dict, layerspec):
    layerfunc = AnnDataDictDataset._parse_layer_spec(layerspec, anndata_dict)

    layers = defaultdict(list)
    for group_name, group in anndata_dict.items():
        for view_name in group.keys():
            layers[view_name].append(layerfunc(group_name, view_name))

    for view_name, view_layers in layers.items():
        assert len(view_layers) == 2
        assert all(view_layers[0] == layer for layer in view_layers[1:])
        view_layer = view_layers[0]
        view_spec = layerspec.get(view_name)
        if not isinstance(view_spec, Mapping):
            assert view_layer == {None: view_spec}
        else:
            assert view_layer == view_spec


@pytest.mark.parametrize("layerspec", _LAYERSPECS_SEPARATE_PER_GROUP_VIEW)
def test_parse_layer_spec_separate_per_group_view(anndata_dict, layerspec):
    layerfunc = AnnDataDictDataset._parse_layer_spec(layerspec, anndata_dict)

    layers = defaultdict(dict)
    for group_name, group in anndata_dict.items():
        for view_name in group.keys():
            layers[view_name][group_name] = layerfunc(group_name, view_name)

    for view_name, view_layers in layers.items():
        assert len(view_layers) == 2
        for group_name, group_layers in view_layers.items():
            cspec = layerspec[group_name].get(view_name)
            if not isinstance(cspec, Mapping):
                assert group_layers == {None: cspec}
            else:
                assert group_layers == cspec


@pytest.mark.parametrize(
    ("layerspec", "exception_type", "exception_match"),
    (
        ("layer4", ValueError, "Layer layer4 not found in group"),
        ({"view_0": {"x1": None, "x2": "layer1"}, "view_1": {"x3": "layer4"}}, ValueError, "Layer layer4 not found"),
        ({"view_0": 1}, TypeError, "Unsupported layer type"),
        ({"x1": None, "x2": "layer4", "x3": "layer1"}, ValueError, "Layer layer4 not found"),
        (
            {"group_4": {"view_0": None}, "group_1": {"view_0": "layer1"}},
            ValueError,
            "Layer specifies group group_4, but it does not exist.",
        ),
        (
            {"group_0": {"view_0": "layer1"}, "group_1": {"view_4": {"x1": None, "x2": "layer1"}}},
            ValueError,
            "Layer specifies view view_4 for group group_1, but it does not exist.",
        ),
        (
            {"group_0": {"view_0": "layer1"}, "group_1": {"view_1": "layer4"}},
            ValueError,
            "Layer layer4 not found in group group_1 for view view_1.",
        ),
        ({"group_0": {"view_0": 1}, "group_1": {"view_1": None}}, TypeError, "Unsupported layer type"),
        (
            {"group_0": {"view_0": None}, "group_1": {"view_0": {"x1": None, "x2": "layer1"}}},
            ValueError,
            "number of specified layers",
        ),
        (
            {"group_0": {"view_0": {"x1": None}}, "group_1": {"view_0": {"x1": None, "x2": "layer1"}}},
            ValueError,
            "number of specified layers",
        ),
        (
            {
                "group_0": {"view_0": None, "view_1": {"x1": "layer1", "x2": "layer2"}},
                "group_1": {"view_0": None, "view_1": {"x1": "layer1"}},
            },
            ValueError,
            "number of specified layers",
        ),
        (
            {
                "group_0": {"view_0": {"x1": None, "x2": "layer1"}},
                "group_1": {"view_0": {"x2": "layer1", "x3": "layer3"}},
            },
            ValueError,
            "assignment of layers to names",
        ),
        ({"x1": "layer1", "x2": "layer4"}, ValueError, "Layer layer4 not found"),
        ({"group_0": {"view_0": None}, "x1": "layer1"}, TypeError, "Unsupported layer type"),
        (1, TypeError, "Unsupported layer type"),
    ),
)
def test_parse_layer_spec_raises(anndata_dict, layerspec, exception_type, exception_match):
    with pytest.raises(exception_type, match=exception_match):
        AnnDataDictDataset._parse_layer_spec(layerspec, anndata_dict)


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


def test_getitems(anndata_dict, dataset, layer, rng, nonmissing_to_slice):
    layerfunc = dataset._parse_layer_spec(layer, anndata_dict)

    anndata_dict = {
        group_name: {
            view_name: AnnData(
                layers={
                    k: view.layers[v] if v is not None else view.X for k, v in layerfunc(group_name, view_name).items()
                },
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
            feature_names = dataset.feature_names[view_name]
            cadata = anndata_dict[group_name][view_name]

            cobsidx = np.isin(sample_names, cadata.obs_names)
            cvaridx = np.isin(feature_names, cadata.var_names)

            cobs = sample_names[cobsidx]
            cvar = feature_names[cvaridx]

            cadata = cadata[cobs, cvar]

            assert cadata.layers.keys() == view.keys()
            for layer_name, clayer in cadata.layers.items():
                cview = view[layer_name]
                assert cview.dtype == np.float32
                if not sparse.issparse(clayer):
                    assert isinstance(cview, np.ndarray)
                else:
                    assert sparse.issparse(cview)
                    clayer = clayer.toarray()
                    cview = cview.toarray()

                assert np.all(clayer == cview)

            cnonmissing_obs = np.nonzero(cobsidx)[0]
            cnonmissing_var = np.nonzero(cvaridx)[0]
            assert np.all(
                nonmissing_to_slice(items["nonmissing_samples"][group_name][view_name], cnonmissing_obs.size)
                == nonmissing_to_slice(cnonmissing_obs, cnonmissing_obs.size)
            )
            assert np.all(
                nonmissing_to_slice(items["nonmissing_features"][group_name][view_name], cnonmissing_var.size)
                == nonmissing_to_slice(cnonmissing_var, cnonmissing_var.size)
            )


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


def test_reindex_samples(anndata_dict, dataset, layer, rng, nonmissing_to_slice):
    samples1, samples2 = {}, {}
    for group_name, group_samples in dataset.sample_names.items():
        selection = rng.choice([True, False], size=len(group_samples), p=[0.5, 0.5])
        samples1[group_name] = group_samples[selection]
        samples2[group_name] = np.concatenate((group_samples[~selection], group_samples[selection][:2]))

    for samples in (samples1, samples2):
        dataset.reindex_samples(samples)
        for group_name, group_samples in samples.items():
            assert np.all(dataset.sample_names[group_name] == group_samples)
        test_getitems(anndata_dict, dataset, layer, rng, nonmissing_to_slice)


def test_reindex_features(anndata_dict, dataset, layer, rng, nonmissing_to_slice):
    features1, features2 = {}, {}
    for view_name, view_features in dataset.feature_names.items():
        selection = rng.choice([True, False], size=len(view_features), p=[0.5, 0.5])
        features1[view_name] = view_features[selection]
        features2[view_name] = np.concatenate((view_features[~selection], view_features[selection][:2]))

    for features in (features1, features2):
        dataset.reindex_features(features)
        for view_name, view_features in features.items():
            assert np.all(dataset.feature_names[view_name] == view_features)
        test_getitems(anndata_dict, dataset, layer, rng, nonmissing_to_slice)
