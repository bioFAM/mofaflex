from collections.abc import Mapping

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
from mofaflex._core.datasets import MofaFlexDataset, MuDataAxis0Dataset, MuDataAxis1Dataset
from mofaflex._core.datasets.mudatadataset import MuDataDataset

_LAYERSPECS_SAME_FOR_ALL = (None, "layer1", {"x1": "layer1", "x2": None})
_LAYERSPECS_PER_MODALITY = (
    {"mod_0": None},
    {"mod_1": "layer2"},
    {"mod_0": "layer1", "mod_1": None, "mod_2": "layer1"},
    {"mod_0": {"x1": "layer1", "x2": None}, "mod_1": None, "mod_2": "layer1"},
)


@pytest.fixture(scope="module", params=(*_LAYERSPECS_SAME_FOR_ALL, *_LAYERSPECS_PER_MODALITY))
def layer(request):
    return request.param


@pytest.fixture(scope="module", params=(None, "global_highly_variable", "highly_variable"))
def subset_var(request):
    return request.param


class MuDataDatasetTest:
    @pytest.fixture(scope="class")
    def mdata(self, rng, mdata_axis):
        nobs = 500
        nvar_per_mod = 20
        ngroups = 4

        obs_names = [f"cell_{i}" for i in range(nobs)]
        var_names = [f"var_{i}" for i in range(nobs)]
        if mdata_axis == 0:
            batch_prefix = "group"
        else:
            batch_prefix = "view"
        adatas = {}
        for mod in range(3):
            if mdata_axis == 0:
                cobs_names = rng.choice(obs_names, size=int(0.8 * nobs), replace=False)
                cvar_names = [f"mod_{mod}_feature_{i}" for i in range(nvar_per_mod)]
            else:
                cobs_names = [f"mod_{mod}_sample_{i}" for i in range(nvar_per_mod)]
                cvar_names = rng.choice(var_names, size=int(0.8 * nobs), replace=False)

            adata = ad.AnnData(
                X=rng.poisson(0.5, size=(len(cobs_names), len(cvar_names))),
                layers={"layer1": rng.normal(0, 1, size=(len(cobs_names), len(cvar_names))).astype(np.float32)},
                obs=pd.DataFrame(index=cobs_names),
                var=pd.DataFrame(index=cvar_names),
            )
            if mod < 2:
                adata.layers["layer2"] = rng.normal(0, 1, size=(len(cobs_names), len(cvar_names))).astype(np.float32)
                adata.obs["covar"] = rng.random(size=adata.n_obs)
                adata.obsm["covar_df"] = pd.DataFrame(
                    rng.random(size=(adata.n_obs, 3)), columns=["a", "b", "c"], index=adata.obs_names
                )
                adata.obsm["covar_array"] = rng.random(size=(adata.n_obs, 3))
                adata.obsm["covar_sparse"] = sparse.csr_array(rng.poisson(size=(adata.n_obs, 3)))
                adata.var["covar"] = rng.random(size=adata.n_vars)
                adata.varm["annot_df"] = pd.DataFrame(
                    rng.random(size=(adata.n_vars, 10)),
                    columns=[f"annot_view_{mod}_{i}" for i in range(10)],
                    index=adata.var_names,
                )
                adata.varm["annot_array"] = rng.random(size=(adata.n_vars, 10))
                adata.varm["annot_sparse"] = sparse.csr_array(rng.poisson(size=(adata.n_vars, 4)))
            adatas[f"mod_{mod}"] = adata

        adatas["mod_0"].X = sparse.csr_array(adatas["mod_0"].X)
        adatas["mod_2"].X = sparse.csc_array(adatas["mod_2"].X)
        adatas["mod_1"].var["highly_variable"] = rng.choice((True, False), size=adata.n_vars, p=[0.4, 0.6])

        mdata = md.MuData(adatas, axis=mdata_axis)

        mdata.obs["batch"] = pd.Categorical(
            rng.choice([f"{batch_prefix}_{i}" for i in range(ngroups)], size=mdata.n_obs).astype(str)
        )
        mdata.var["batch"] = pd.Categorical(
            rng.choice([f"{batch_prefix}_{i}" for i in range(ngroups)], size=mdata.n_var).astype(str)
        )

        global_covar = rng.random(size=mdata.n_obs)
        global_covar[~mdata.obs_names.isin(adatas["mod_2"].obs_names)] = np.nan
        mdata.obs["covar"] = global_covar

        global_covar = rng.random(size=mdata.n_vars)
        global_covar[~mdata.var_names.isin(adatas["mod_2"].var_names)] = np.nan
        mdata.var["covar"] = global_covar

        global_covar = rng.random(size=(mdata.n_obs, 3))
        global_covar[~mdata.obs_names.isin(adatas["mod_2"].obs_names)] = np.nan
        mdata.obsm["covar_df"] = pd.DataFrame(global_covar, columns=["a", "b", "c"], index=mdata.obs_names)
        mdata.obsm["covar_array"] = rng.permuted(global_covar, axis=0)

        global_covar_sparse = sparse.csr_array(rng.poisson(size=(mdata.n_obs, 3)).astype(np.float32))
        global_covar_sparse[~mdata.obs_names.isin(adatas["mod_2"].obs_names)] = np.nan
        mdata.obsm["covar_sparse"] = global_covar_sparse

        global_annot = rng.random(size=(mdata.n_vars, 10))
        global_annot[~mdata.var_names.isin(adatas["mod_2"].var_names)] = np.nan
        mdata.varm["annot_df"] = pd.DataFrame(
            global_annot, columns=[f"global_annot_{i}" for i in range(10)], index=mdata.var_names
        )
        mdata.varm["annot_array"] = rng.permuted(global_annot, axis=0)

        global_annot_sparse = sparse.csr_array(rng.poisson(size=(mdata.n_var, 4)).astype(np.float32))
        global_annot_sparse[~mdata.var_names.isin(adatas["mod_2"].var_names)] = np.nan
        mdata.varm["annot_sparse"] = global_annot_sparse

        mdata.var["global_highly_variable"] = rng.choice((True, False), size=mdata.n_vars, p=[0.3, 0.7])

        return mdata

    @pytest.fixture(scope="function")
    def dataset(self, mdata, layer, subset_var):
        return MofaFlexDataset(mdata, group_by="batch", layer=layer, subset_var=subset_var, cast_to=np.float32)

    def test_properties(self, mdata, dataset, subset_var):
        assert sorted(dataset.group_names) == sorted(self.get_group_names(mdata))
        assert sorted(dataset.view_names) == sorted(self.get_view_names(mdata))
        for group_name, sample_names in dataset.sample_names.items():
            cmdata = self.get_mdata_subset(mdata, group_name=group_name)
            assert np.all(np.sort(sample_names) == cmdata.obs_names.sort_values().to_numpy())
            assert dataset.n_samples[group_name] == cmdata.n_obs

        for view_name, view_names in dataset.feature_names.items():
            cvarnames = self.get_varnames(mdata, subset_var, view_name=view_name)

            assert np.all(view_names == cvarnames)
            assert dataset.n_features[view_name] == cvarnames.size

    @pytest.mark.parametrize("layerspec", _LAYERSPECS_SAME_FOR_ALL)
    def test_parse_layer_spec_same_for_all(self, mdata, layerspec):
        layerfunc = MuDataDataset._parse_layer_spec(layerspec, mdata)

        layers = [layerfunc(modname) for modname in mdata.mod.keys()]
        assert all(layers[0] == layer for layer in layers[1:])
        layer = layers[0]
        if not isinstance(layerspec, Mapping):
            assert layer == {None: layerspec}
        else:
            assert layer == layerspec

    @pytest.mark.parametrize("layerspec", _LAYERSPECS_PER_MODALITY)
    def test_parse_layer_spec_per_modality(self, mdata, layerspec):
        layerfunc = MuDataDataset._parse_layer_spec(layerspec, mdata)

        layers = {modname: layerfunc(modname) for modname in mdata.mod.keys()}
        for modname in mdata.mod.keys():
            if not isinstance(cspec := layerspec.get(modname), Mapping):
                assert layers[modname] == {None: cspec}
            else:
                assert layers[modname] == cspec

    @pytest.mark.parametrize(
        ("layerspec", "exception_type", "exception_match"),
        (
            ("layer4", ValueError, "Layer layer4 not found"),
            ({"mod_0": "layer4"}, ValueError, "Layer layer4 not found in modality mod_0."),
            ({"mod_0": {"x1": "layer1", "x2": "layer4"}}, ValueError, "Layer layer4 not found in modality mod_0."),
            ({"mod_0": {"x1": "layer1", "x2": 1}}, TypeError, "Unsupported layer type"),
            ({"mod_0": 1}, TypeError, "Unsupported layer type"),
            ({"x1": "layer1", "x2": "layer4"}, ValueError, "Layer layer4 not found"),
            ({"x1": "layer1", "x2": 1}, TypeError, "Unsupported layer type"),
            (1, TypeError, "Unsupported layer type"),
        ),
    )
    def test_parse_layer_spec_raises(self, mdata, layerspec, exception_type, exception_match):
        with pytest.raises(exception_type, match=exception_match):
            MuDataDataset._parse_layer_spec(layerspec, mdata)

    @pytest.mark.parametrize("axis", (0, 1, 2))
    def test_alignment(self, mdata, dataset, rng, axis):
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

                mod = self.get_mdata_subset(mdata, group_name=group_name, view_name=view_name)
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

                mod = self.get_mdata_subset(mdata, group_name=group_name, view_name=view_name)
                local_varnames = view_features[np.isin(view_features, mod.var_names)]
                idx = pd.Index(view_features).get_indexer(local_varnames)
                assert np.all(local_arr == np.take(global_arr, idx, axis=axis))

                idx = np.isin(view_features, local_varnames)
                assert np.all(np.isnan(np.compress(~idx, new_global_arr, axis=axis)))
                assert np.all(np.compress(idx, new_global_arr, axis=axis) == np.compress(idx, global_arr, axis=axis))

    def test_index_mapping(self, mdata, dataset, rng):
        for group_name, group_samples in dataset.sample_names.items():
            global_idx = rng.choice(group_samples.size, size=int(0.3 * group_samples.size), replace=True)
            for view_name in dataset.view_names:
                local_idx = dataset.map_global_indices_to_local(global_idx, group_name, view_name, align_to="samples")

                local_obsnames = self.get_mdata_subset(
                    mdata, group_name=group_name, view_name=view_name
                ).obs_names.intersection(group_samples, sort=False)
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

                local_varnames = self.get_mdata_subset(
                    mdata, group_name=group_name, view_name=view_name
                ).var_names.intersection(view_features, sort=False)
                assert np.all(view_features[global_idx][local_idx >= 0] == local_varnames[local_idx[local_idx >= 0]])
                assert np.all(~np.isin(view_features[global_idx][local_idx < 0], local_varnames))

                new_global_idx = dataset.map_local_indices_to_global(
                    local_idx[local_idx >= 0], group_name, view_name, align_to="features"
                )
                assert np.all(global_idx[local_idx >= 0] == new_global_idx)

    def test_getitems(self, mdata, dataset, layer, rng, nonmissing_to_slice):
        layerfunc = dataset._parse_layer_spec(layer, mdata)
        mods = {}
        for modname in mdata.mod.keys():
            adata = mdata.mod[modname]
            mods[modname] = AnnData(
                layers={k: adata.layers[v] if v is not None else adata.X for k, v in layerfunc(modname).items()},
                obs=adata.obs,
                var=adata.var,
                obsm=adata.obsm,
                varm=adata.varm,
            )
        new_mdata = MuData(
            mods, obs=mdata.obs, var=mdata.var, obsmap=mdata.obsmap, varmap=mdata.varmap, axis=mdata.axis
        )
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
                feature_names = dataset.feature_names[view_name]
                cdata = self.get_mdata_subset(
                    mdata[sample_names, feature_names], group_name=group_name, view_name=view_name
                )

                assert cdata.layers.keys() == view.keys()
                for layer_name, clayer in cdata.layers.items():
                    cview = view[layer_name]
                    assert cview.dtype == np.float32
                    if not sparse.issparse(clayer):
                        assert isinstance(cview, np.ndarray)
                    else:
                        assert sparse.issparse(cview)
                        clayer = clayer.toarray()
                        cview = cview.toarray()

                    assert np.all(clayer == cview)

                cnonmissing_obs = nonmissing_to_slice(
                    np.nonzero(np.isin(sample_names, cdata.obs_names))[0], sample_names.size
                )
                assert np.all(
                    nonmissing_to_slice(items["nonmissing_samples"][group_name][view_name], sample_names.size)
                    == cnonmissing_obs
                )

                cnonmissing_var = nonmissing_to_slice(
                    np.nonzero(np.isin(feature_names, cdata.var_names))[0], feature_names.size
                )
                assert np.all(
                    nonmissing_to_slice(items["nonmissing_features"][group_name][view_name], feature_names.size)
                    == cnonmissing_var
                )

    @pytest.mark.parametrize("axis", [0, 1])
    def test_get_covariates_from_key(self, mdata, dataset, subset_var, axis):
        attr = "obs" if axis == 0 else "var"
        namesattr = f"{attr}_names"
        dsetnamesattr = "sample_names" if axis == 0 else "feature_names"
        dict_reorder = slice(None) if axis == 0 else slice(None, None, -1)

        covars = dataset.get_covariates(axis=axis, key="covar")
        outer_iterator, inner_value, subset_reorder, get_subdata = self.get_covariates_from_key_params(dataset)

        for outer_name in outer_iterator:
            subdata = get_subdata(mdata, outer_name)
            for modname in subdata.mod.keys():
                dict_key = (outer_name, modname)[subset_reorder][dict_reorder]
                assert covars[dict_key[0]][dict_key[1]].columns == ["covar"]

                names = getattr(dataset, dsetnamesattr)[dict_key[0]]
                if modname != inner_value:
                    mod = subdata.mod[modname][:, self.get_varnames(subdata, subset_var, outer_name, modname)]
                    globalidx = np.isin(names, getattr(mod, namesattr))
                    localidx = getattr(mod, namesattr).get_indexer(names)
                    localidx = localidx[localidx >= 0]
                    assert np.all(
                        covars[dict_key[0]][dict_key[1]].iloc[globalidx, 0].squeeze()
                        == getattr(mod, attr)["covar"].to_numpy()[localidx]
                    )
                    assert np.all(np.isnan(covars[dict_key[0]][dict_key[1]].iloc[~globalidx, 0]))
                else:
                    covar = getattr(subdata, attr).loc[names, "covar"].to_numpy()
                    nanidx = np.isnan(covar)
                    assert np.all(covar[~nanidx] == covars[dict_key[0]][dict_key[1]].iloc[~nanidx, 0].squeeze())
                    assert np.all(np.isnan(covars[dict_key[0]][dict_key[1]].iloc[nanidx, 0]))

    @pytest.mark.parametrize("type", ("df", "array", "sparse"))
    @pytest.mark.parametrize("axis, mkey", [(0, "covar"), (1, "annot")])
    def test_get_covariates_from_keym(self, mdata, dataset, subset_var, axis, mkey, type):
        mkey = f"{mkey}_{type}"
        attr = "obs" if axis == 0 else "var"
        attrm = f"{attr}m"
        namesattr = f"{attr}_names"
        dsetnamesattr = "sample_names" if axis == 0 else "feature_names"
        dict_reorder = slice(None) if axis == 0 else slice(None, None, -1)

        covars = dataset.get_covariates(axis=axis, mkey=mkey)
        outer_iterator, inner_value, subset_reorder, _ = self.get_covariates_from_key_params(dataset)

        for outer_name in outer_iterator:
            for modname in mdata.mod.keys():
                group_name, view_name = (outer_name, modname)[subset_reorder]
                dict_key = (group_name, view_name)[dict_reorder]
                names = getattr(dataset, dsetnamesattr)[dict_key[0]]
                subdata = mdata[dataset.sample_names[group_name], dataset.feature_names[view_name]]
                ccovars = covars[dict_key[0]][dict_key[1]]
                if modname != inner_value:
                    view = subdata.mod[modname]

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

    def test_get_missing_obs(self, mdata, dataset):
        missing = dataset.get_missing_obs()
        for (group_name, view_name), df in missing.groupby(["group", "view"]):
            view = self.get_mdata_subset(mdata, group_name, view_name)
            cmissing = ~np.isin(dataset.sample_names[group_name], view.obs_names)
            df = df.set_index("obs_name")
            assert np.all(df.loc[dataset.sample_names[group_name], "missing"][cmissing])
            assert np.all(~df.loc[dataset.sample_names[group_name], "missing"][~cmissing])

    def test_reindex_samples(self, mdata, dataset, layer, rng, nonmissing_to_slice):
        samples1, samples2 = {}, {}
        for group_name, group_samples in dataset.sample_names.items():
            selection = rng.choice([True, False], size=len(group_samples), p=[0.5, 0.5])
            samples1[group_name] = group_samples[selection]
            samples2[group_name] = np.concatenate((group_samples[~selection], group_samples[selection][:2]))

        for samples in (samples1, samples2):
            dataset.reindex_samples(samples)
            for group_name, group_samples in samples.items():
                assert np.all(dataset.sample_names[group_name] == group_samples)
            self.test_getitems(mdata, dataset, layer, rng, nonmissing_to_slice)

    def test_reindex_features(self, mdata, dataset, layer, rng, nonmissing_to_slice):
        features1, features2 = {}, {}
        for view_name, view_features in dataset.feature_names.items():
            selection = rng.choice([True, False], size=len(view_features), p=[0.5, 0.5])
            features1[view_name] = view_features[selection]
            features2[view_name] = np.concatenate((view_features[~selection], view_features[selection][:2]))

        for features in (features1, features2):
            dataset.reindex_features(features)
            for view_name, view_features in features.items():
                assert np.all(dataset.feature_names[view_name] == view_features)
            self.test_getitems(mdata, dataset, layer, rng, nonmissing_to_slice)


class TestMuDataAxis0Dataset(MuDataDatasetTest):
    @pytest.fixture(scope="class")
    def mdata_axis(self, request):
        return 0

    @staticmethod
    def get_varnames(mdata, subset_var, group_name=None, view_name=None):
        varnames = mdata.mod[view_name].var_names
        if subset_var in mdata.var:
            varnames = varnames[mdata.var[subset_var][mdata.varmap[view_name].reshape(-1) > 0]]
        elif subset_var in mdata[view_name].var:
            varnames = varnames[mdata[view_name].var[subset_var]]
        return varnames

    @staticmethod
    def get_group_names(mdata):
        return mdata.obs["batch"].unique()

    @staticmethod
    def get_view_names(mdata):
        return mdata.mod.keys()

    @staticmethod
    def get_mdata_subset(mdata, group_name=None, view_name=None):
        if group_name is not None:
            mdata = mdata[mdata.obs["batch"] == group_name, :]
        if view_name is not None:
            mdata = mdata[view_name]
        return mdata

    @staticmethod
    def get_covariates_from_key_params(dataset):
        return (
            dataset.group_names,
            "mod_2",
            slice(None),
            lambda mdata, group_name: mdata[mdata.obs["batch"] == group_name, :],
        )

    def test_instance(self, dataset, mdata_axis):
        assert isinstance(dataset, MuDataAxis0Dataset)

    @pytest.mark.parametrize("usedask", [False, True])
    def test_apply_by_group_view(self, mdata, dataset, usedask):
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
    def test_apply_by_view(self, mdata, dataset, usedask):
        def applyfun(adata, group_name, view_name):
            assert np.all(adata.obs_names.sort_values() == mdata[view_name].obs_names.sort_values())
            assert np.all(adata.var_names == dataset.feature_names[view_name])

        with settings.override(use_dask=usedask):
            dataset.apply(applyfun, by_group=False)

    @pytest.mark.xfail(
        Version(ad.__version__) < Version("0.11.4"), reason="anndata bug: https://github.com/scverse/anndata/pull/1911"
    )
    @pytest.mark.parametrize("usedask", [False, True])
    def test_apply_by_group(self, mdata, dataset, usedask):
        varnames = np.concat(tuple(dataset.feature_names.values()))

        def applyfun(adata, group_name, view_name):
            assert np.all(adata.obs_names == dataset.sample_names[group_name])
            assert np.all(adata.var_names == varnames)

        with settings.override(use_dask=usedask):
            dataset.apply(applyfun, by_view=False)

    @pytest.mark.parametrize("usedask", [False, True])
    def test_apply_to_view(self, mdata, dataset, usedask):
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
    def test_apply_to_group(self, mdata, dataset, usedask):
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


class TestMuDataAxis1Dataset(MuDataDatasetTest):
    @pytest.fixture(scope="class")
    def mdata_axis(self, request):
        return 1

    @staticmethod
    def get_varnames(mdata, subset_var, view_name=None, group_name=None):
        mask = mdata.var["batch"] == view_name
        if subset_var in mdata.var:
            mask &= mdata.var[subset_var]
        else:
            for modname, mod in mdata.mod.items():
                if subset_var in mod.var:
                    map = mdata.varmap[modname].ravel()
                    mapmask = map > 0
                    cmask = np.zeros_like(mask)
                    cmask[mapmask] = mod.var[subset_var].iloc[map[mapmask] - 1]
                    mask &= cmask
        if group_name is not None:
            mapmask = mdata.varmap[group_name].ravel()
            return mdata.var_names[mapmask > 0 & mask]
        else:
            return mdata.var_names[mask]

    @staticmethod
    def get_group_names(mdata):
        return mdata.mod.keys()

    @staticmethod
    def get_view_names(mdata):
        return mdata.var["batch"].unique()

    @staticmethod
    def get_mdata_subset(mdata, group_name=None, view_name=None):
        if view_name is not None:
            mdata = mdata[:, mdata.var["batch"] == view_name]
        if group_name is not None:
            mdata = mdata[group_name]
        return mdata

    @staticmethod
    def get_covariates_from_key_params(dataset):
        return (
            dataset.view_names,
            "mod_2",
            slice(None, None, -1),
            lambda mdata, view_name: mdata[:, mdata.var["batch"] == view_name],
        )

    def test_instance(self, dataset, mdata_axis):
        assert isinstance(dataset, MuDataAxis1Dataset)

    @pytest.mark.parametrize("usedask", [False, True])
    def test_apply_by_group_view(self, mdata, dataset, usedask):
        def applyfun(adata, group_name, view_name, ref_sample_names, ref_feature_names):
            assert np.all(adata.obs_names == ref_sample_names)
            assert np.all(
                adata.var_names == pd.Index(ref_feature_names).intersection(mdata[group_name].var_names, sort=False)
            )

        with settings.override(use_dask=usedask):
            dataset.apply(
                applyfun,
                group_kwargs={"ref_sample_names": dataset.sample_names},
                view_kwargs={"ref_feature_names": dataset.feature_names},
            )

    @pytest.mark.xfail(
        Version(ad.__version__) < Version("0.11.4"), reason="anndata bug: https://github.com/scverse/anndata/pull/1911"
    )
    @pytest.mark.parametrize("usedask", [False, True])
    def test_apply_by_view(self, mdata, dataset, usedask):
        obsnames = np.concat(tuple(dataset.sample_names.values()))

        def applyfun(adata, group_name, view_name):
            assert np.all(adata.obs_names == obsnames)
            assert np.all(adata.var_names == dataset.feature_names[view_name])

        with settings.override(use_dask=usedask):
            dataset.apply(applyfun, by_group=False)

    @pytest.mark.parametrize("usedask", [False, True])
    def test_apply_by_group(self, mdata, dataset, usedask):
        def applyfun(adata, group_name, view_name):
            assert np.all(adata.obs_names == dataset.sample_names[group_name])

            for vname in np.unique(view_name):
                assert np.all(
                    adata.var_names[view_name == vname].sort_values()
                    == mdata[group_name].var_names.intersection(dataset.feature_names[vname]).sort_values()
                )

        with settings.override(use_dask=usedask):
            dataset.apply(applyfun, by_view=False)

    @pytest.mark.parametrize("usedask", [False, True])
    def test_apply_to_view(self, mdata, dataset, usedask):
        def applyfun(adata, group_name, ref_sample_names, ref_feature_names):
            assert np.all(adata.obs_names == ref_sample_names)
            assert np.all(
                adata.var_names == pd.Index(ref_feature_names).intersection(mdata[group_name].var_names, sort=False)
            )

        with settings.override(use_dask=usedask):
            for view_name in dataset.view_names:
                dataset.apply_to_view(
                    view_name,
                    applyfun,
                    group_kwargs={"ref_sample_names": dataset.sample_names},
                    ref_feature_names=dataset.feature_names[view_name],
                )

    @pytest.mark.parametrize("usedask", [False, True])
    def test_apply_to_group(self, mdata, dataset, usedask):
        def applyfun(adata, group_name, ref_sample_names, ref_feature_names, _group_name):
            assert np.all(adata.obs_names == ref_sample_names)
            assert np.all(
                adata.var_names == pd.Index(ref_feature_names).intersection(mdata[_group_name].var_names, sort=False)
            )

        with settings.override(use_dask=usedask):
            for group_name in dataset.group_names:
                dataset.apply_to_group(
                    group_name,
                    applyfun,
                    view_kwargs={"ref_feature_names": dataset.feature_names},
                    ref_sample_names=dataset.sample_names[group_name],
                    _group_name=group_name,
                )
