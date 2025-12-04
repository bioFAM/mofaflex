import logging

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats import multitest

from .datasets import MofaFlexDataset
from .utils import sample_all_data_as_one_batch

_logger = logging.getLogger(__name__)


def _test_single_view(
    view_name: str,
    nonnegative_weights: bool,
    feature_sets: pd.DataFrame,
    factor_loadings: pd.DataFrame,
    y: pd.DataFrame | None,
    sign: str = "all",
    corr_adjust: bool = True,
    p_adj_method: str = "fdr_bh",
    min_size: int = 10,
    subsample: int = 0,
):
    """Perform significance test of factor loadings against feature sets.

    Args:
        view_name: View name.
        nonnegative_weights: Whether the model was constrained to nonnegative weights for this view.
        feature_sets: Annotations.
        factor_loadings: Weights.
        y: The data. Only required if `corr_adjust=True`.
        sign: Test direction - "all" for two-sided, "neg" or "pos" for one-sided tests.
        corr_adjust: Whether to adjust for correlations between features.
        p_adj_method: Method for multiple testing adjustment (e.g. "fdr_bh").
        min_size: Minimum size threshold for feature sets.
        subsample: Work with a random subsample of the data to speed up testing, 0 means no subsampling.

    Returns:
        dict: Test results containing:
            - "t": DataFrame of t-statistics
            - "p": DataFrame of p-values
            - "padj": DataFrame of adjusted p-values (if p_adj_method is not None)
    """
    if nonnegative_weights and sign == "neg":
        return None

    if not feature_sets.any(axis=None):
        return None
    feature_sets = feature_sets.loc[:, feature_sets.sum(axis=0) >= min_size]

    if not feature_sets.any(axis=None):
        _logger.warning(f"No feature sets with more than {min_size} features for view {view_name}, skipping view.")
        return None

    feature_sets = feature_sets.loc[feature_sets.any(axis=1), ~(feature_sets.all(axis=0))]
    if not feature_sets.any(axis=None):
        _logger.warning(f"No feature sets with unique annotations for view {view_name}, skipping view.")
        return None

    factor_loadings = factor_loadings / np.max(np.abs(factor_loadings.to_numpy()))

    if sign == "pos":
        factor_loadings[factor_loadings < 0] = 0.0
    if sign == "neg":
        factor_loadings[factor_loadings > 0] = 0.0
    factor_loadings = factor_loadings.abs()

    factor_names = factor_loadings.columns

    t_stat_dict = {}
    prob_dict = {}

    for feature_set in feature_sets.columns:
        fs_features = feature_sets[feature_set]
        features_in = fs_features.index[fs_features]
        features_out = fs_features.index[~fs_features]

        loadings_in = factor_loadings.loc[features_in, :]
        loadings_out = factor_loadings.loc[features_out, :]

        n_in = features_in.shape[0]
        n_out = features_out.shape[0]

        df = n_in + n_out - 2.0
        mean_diff = loadings_in.mean(axis=0) - loadings_out.mean(axis=0)
        # why divide here by df and not denom later?
        svar = ((n_in - 1) * loadings_in.var(axis=0) + (n_out - 1) * loadings_out.var(axis=0)) / df

        vif = 1.0
        if corr_adjust:
            corr_df = y.loc[:, features_in].corr()
            mean_corr = (np.nansum(corr_df.to_numpy()) - n_in) / (n_in * (n_in - 1))
            vif = 1 + (n_in - 1) * mean_corr
            df = y.shape[0] - 2
        denom = np.sqrt(svar * (vif / n_in + 1.0 / n_out))

        with np.errstate(divide="ignore", invalid="ignore"):
            t_stat = mean_diff / denom
        prob = t_stat.apply(lambda t: stats.t.sf(np.abs(t), df) * 2)  # noqa B023

        t_stat_dict[feature_set] = t_stat
        prob_dict[feature_set] = prob

    t_stat_df = pd.DataFrame(t_stat_dict, index=factor_names)
    prob_df = pd.DataFrame(prob_dict, index=factor_names)
    t_stat_df.fillna(0.0, inplace=True)
    prob_df.fillna(1.0, inplace=True)

    adjust_p = p_adj_method is not None
    if adjust_p:
        prob_adj_df = prob_df.apply(
            lambda p: multitest.multipletests(p, method=p_adj_method)[1], axis=1, result_type="broadcast"
        )

    if sign != "all":
        prob_df[t_stat_df < 0.0] = 1.0
        if adjust_p:
            prob_adj_df[t_stat_df < 0.0] = 1.0
        t_stat_df[t_stat_df < 0.0] = 0.0

    result = {"t": t_stat_df, "p": prob_df}
    if adjust_p:
        result["padj"] = prob_adj_df

    return result


def pcgse_test(
    data: MofaFlexDataset | None,
    nonnegative_weights: dict[str, bool],
    annotations: dict[str, pd.DataFrame],
    weights: dict[str, pd.DataFrame],
    corr_adjust: bool = True,
    p_adj_method: str = "fdr_bh",
    min_size: int = 10,
    subsample: int = 0,
):
    """Perform significance testing across multiple views and sign directions.

    Args:
        data: A MOFA-FLEX dataset.
        nonnegative_weights: Whether the model was constrained to nonnegative weights for each view.
        annotations: Boolean dataframe with feature sets in each row for each view.
        weights: Weights for each view.
        corr_adjust: Whether to adjust for correlations between features.
        p_adj_method: Method for multiple testing adjustment.
        min_size: Minimum size threshold for feature sets.
        subsample: Work with a random subsample of the data to speed up testing.

    Returns:
        Test results for each view, where test results contain t-statistics, p-values, and adjusted p-values
            separately for each side of a one-sided test.
    """
    results = {}

    y = None
    if corr_adjust:
        if subsample is not None and subsample > 0:
            idx = {
                group_name: (
                    np.random.choice(nsamples, subsample, replace=False)
                    if subsample < nsamples
                    else np.arange(nsamples)
                )
                for group_name, nsamples in data.n_samples.items()
            }
        else:
            idx = sample_all_data_as_one_batch(data)
        y = data.__getitems__(idx)["data"]
        y = {
            view_name: pd.DataFrame(
                np.concatenate(
                    [
                        data.align_local_array_to_global(
                            group[view_name], group_name, view_name, align_to="features", axis=1
                        )
                        for group_name, group in y.items()
                    ],
                    axis=0,
                ),
                columns=data.feature_names[view_name],
            )
            for view_name in data.view_names
        }

    for view_name, loadings in weights.items():
        if view_name in annotations:
            view_results = []
            for sign in ["neg", "pos"]:
                cresult = _test_single_view(
                    view_name=view_name,
                    nonnegative_weights=nonnegative_weights[view_name],
                    feature_sets=annotations[view_name],
                    factor_loadings=loadings,
                    y=y[view_name] if y is not None else None,
                    sign=sign,
                    corr_adjust=corr_adjust,
                    p_adj_method=p_adj_method,
                    min_size=min_size,
                    subsample=subsample,
                )
                if cresult is not None:
                    dfs = [
                        df.melt(var_name="annotation", value_name=name, ignore_index=False)
                        .rename_axis(index="factor")
                        .set_index("annotation", append=True)
                        for name, df in cresult.items()
                    ]
                    view_results.append(pd.concat(dfs, axis=1).reset_index(drop=False).assign(sign=sign))
            results[view_name] = pd.concat(view_results, axis=0, ignore_index=True)
    return results
