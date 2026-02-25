from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import suppress
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotnine as p9
from anndata import AnnData
from mizani import bounds
from mizani.palettes import brewer_pal
from mudata import MuData

from .. import tl

if TYPE_CHECKING:
    from .._core import MOFAFLEX, MofaFlexDataset
    from .._core.api import types


def _rescale_zerosymmetric(x, to: tuple[float, float] = (0, 1), _from: tuple[float, float] | None = None):
    _from = _from or (np.min(x), np.max(x))
    return np.interp(x, (_from[0], 0, _from[1]), (0, 0.5, 1))


_scale_fill_zerosymmetric_diverging = partial(
    p9.scale_fill_gradientn,
    colors=brewer_pal(type="div", palette="RdBu", direction=-1)(11),
    rescaler=_rescale_zerosymmetric,
    expand=(0, 0),
)

_scale_color_zerosymmetric_diverging = partial(
    p9.scale_color_gradientn,
    colors=brewer_pal(type="div", palette="RdBu", direction=-1)(11),
    rescaler=_rescale_zerosymmetric,
    expand=(0, 0),
)

_no_axis_ticks_x = {"axis_ticks_length_major_x": 0, "axis_ticks_length_minor_x": 0}
_no_axis_ticks_y = {"axis_ticks_length_major_y": 0, "axis_ticks_length_minor_y": 0}


def _covariate_df(data, key, axis=0):
    cov = data.get_covariates(axis=axis, key=key)
    for group_name, group in cov.items():
        have_covariate = False
        for view in group.values():
            with suppress(KeyError):
                cov[group_name] = view[key]
                have_covariate = True
                break
        if not have_covariate:  # can't use else here in case group is empty
            raise ValueError(f"Covariate '{key}' not found in the data for group {group_name}.")
    if len(cov) == 0:
        raise ValueError(f"Covariate '{key}' not found in the data.")
    return cov


def factors_scatter(
    model: types.MofaFlex | MOFAFLEX,
    x: int | str,
    y: int | str,
    groups: str | Sequence[str] | None = None,
    color: str | None = None,
    shape: str | None = None,
    data: MuData | Mapping[str, Mapping[str, AnnData]] | AnnData | None = None,
    size: float = 2,
    alpha: float = 1,
    figsize: tuple[float, float] | None = None,
    nrow: int | None = None,
    ncol: int | None = None,
) -> p9.ggplot:
    """Plot two factors against each other and color by covariates.

    Args:
        model: The term to plot the factor correlation for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        x: The factor to plot on the x-axis.
        y: The factor to plot on the y-axis.
            only one additive term.
        groups: The groups to plot. If `None`, all groups are shown.
        color: The covariate name to color by.
        shape: The covariate name to shape by.
        data: The data that the model was trained on. Only required if `color is not None` or `shape is not None`.
        size: Size of the data points.
        alpha: Transparency of the data points.
        figsize: Figure size in inches.
        nrow: Number of rows in the faceted plot. If None, plotnine will determine automatically.
        ncol: Number of columns in the faceted plot. If None, plotnine will determine automatically.
    """
    if isinstance(groups, str):
        groups = [groups]
    elif groups is None:
        groups = model.group_names
    if figsize is None:
        figsize = (5 * len(groups), 5)

    if isinstance(x, int):
        if x < 1 or x > model.n_factors:
            raise ValueError(f"Factor 'x' must be between 1 and {model.n_factors}.")
        else:
            x = model.factor_names[x - 1]
    if isinstance(y, int):
        if y < 1 or y > model.n_factors:
            raise ValueError(f"Factor 'y' must be between 1 and {model.n_factors}.")
        else:
            y = model.factor_names[y - 1]

    if (color is not None or shape is not None) and data is None:
        raise ValueError("'data' cannot be 'None' if 'color' or 'shape' are given.")
    elif data is not None:
        data = model._make_dataset(data)

    df_factors = pd.concat(model.get_factors(), axis=0)

    toconcat = []
    if color is not None:
        toconcat.append(pd.concat(_covariate_df(data, color), axis=0))
    if shape is not None:
        toconcat.append(pd.concat(_covariate_df(data, shape), axis=0))
    if len(toconcat) > 0:
        df_factors = pd.concat((df_factors, *toconcat), axis=1)
    df_factors.reset_index(names=["group", "sample"], inplace=True)

    aes_kwargs = {}
    if color is not None:
        aes_kwargs["color"] = color
    if shape is not None:
        aes_kwargs["shape"] = shape

    plot = (
        p9.ggplot(df_factors, p9.aes(x=x, y=y, **aes_kwargs))
        + p9.geom_hline(yintercept=0, linetype="dashed", color="black")
        + p9.geom_vline(xintercept=0, linetype="dashed", color="black")
        + p9.geom_point(size=size, alpha=alpha, stroke=0)
        + p9.facet_wrap("group", ncol=ncol, nrow=nrow)
        + p9.theme(figure_size=figsize)
    )

    return plot


def _plot_covariates_scatter(
    model: types.MofaFlex | MOFAFLEX,
    axis: Literal[0, 1],
    factor: int | str,
    groups: str | Sequence[str] | None = None,
    covariate_dims: int | str | Sequence[int] | Sequence[str] | None = None,
    color: int | str | None = None,
    shape: str | None = None,
    data: MuData | Mapping[str, Mapping[str, AnnData]] | AnnData | None = None,
    size: float = 1,
    alpha: float = 1,
    figsize: tuple[float, float] = (6, 6),
) -> p9.ggplot:
    if isinstance(factor, int):
        if factor not in range(1, model.n_total_factors + 1):
            raise ValueError(f"Factor must be between 1 and {model.n_total_factors}.")
        else:
            factor = model.factor_names[factor - 1]

    if axis == 0:
        groups_attr = "group_names"
        covariates_attr = "factor_covariates"
    else:
        groups_attr = "view_names"
        covariates_attr = "weight_covariates"
    if isinstance(groups, str):
        groups = [groups]
    elif groups is None:
        groups = [
            group_name for group_name in getattr(model, groups_attr) if group_name in getattr(model, covariates_attr)
        ]
    if figsize is None:
        figsize = (5 * len(groups), 5)

    df_factors = pd.concat(model.get_factors(), axis=0)
    df_covariates = pd.concat(getattr(model, covariates_attr), axis=0)

    if isinstance(covariate_dims, int | str):
        covariate_dims = [covariate_dims]

    if covariate_dims is None:
        covariate_dims = df_covariates.columns
    else:
        covariate_dims = [df_covariates.columns[d] if isinstance(d, int) else d for d in covariate_dims]

    if len(covariate_dims) == 2 and color is not None:
        raise ValueError("Cannot specify a color variable when plotting two covariate dimensions.")
    elif len(covariate_dims) not in (1, 2):
        raise ValueError("Can only plot 1 or 2 covariate dimensions.")

    if (color is not None or shape is not None) and data is None:
        raise ValueError("'data' cannot be 'None' if 'color' or 'shape' are given.")
    elif data is not None:
        data = model._make_dataset(data)

    toconcat = []
    if color is not None:
        if isinstance(color, int):
            color = model.factor_names[color]
        elif color not in df_factors.columns:
            toconcat.append(pd.concat(_covariate_df(data, color, axis), axis=0))
    if shape is not None and shape not in df_factors.columns:
        toconcat.append(pd.concat(_covariate_df(data, shape, axis), axis=0))

    df = pd.concat([df_factors, df_covariates, *toconcat], axis=1).reset_index(names=["group", "sample"])

    aes_kwargs = {}
    if len(covariate_dims) == 1:
        x = covariate_dims[0]
        y = factor
        if color is not None:
            aes_kwargs["color"] = color

    elif len(covariate_dims) == 2:
        x = covariate_dims[0]
        y = covariate_dims[1]
        aes_kwargs["color"] = factor

    if shape is not None:
        aes_kwargs["shape"] = shape

    plot = (
        p9.ggplot(df, p9.aes(x=x, y=y, **aes_kwargs))
        + p9.geom_point(size=size, alpha=alpha, stroke=0)
        + p9.facet_wrap("group")
        + p9.theme(figure_size=figsize)
    )

    return plot


def covariates_factor_scatter(
    model: types.MofaFlex | MOFAFLEX,
    factor: int | str,
    groups: str | Sequence[str] | None = None,
    covariate_dims: int | str | Sequence[int] | Sequence[str] | None = None,
    color: int | str | None = None,
    shape: str | None = None,
    data: MuData | Mapping[str, Mapping[str, AnnData]] | AnnData | None = None,
    size: float = 1,
    alpha: float = 1,
    figsize: tuple[float, float] = (6, 6),
) -> p9.ggplot:
    """Plot a factor against one or two covariate dimensions.

    Args:
        model: The term to plot the factor correlation for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        factor: The factor to plot.
        groups: The groups to plot. If `None`, all groups with covariates are shown.
        covariate_dims: The dimensions of the covariates to plot against. If a list of length 1, plot covariate
            on the x-axis and factor on the y-axis. If a list of length 2, plot the first covariate on the x-axis,
            the second covariate on the y-axis, and factor as color. If None, use all dimensions.
        color: The factor or covariate to color by. Only used when one covariate dimension is plotted.
        shape: The covariate name to shape by.
        data: The data that the model was trained on. Only required if `color is not None` or `shape is not None`.
        size: Size of the data points.
        alpha: Transparency of the data points.
        figsize: Figure size in inches.
    """
    return _plot_covariates_scatter(
        model,
        axis=0,
        factor=factor,
        groups=groups,
        covariate_dims=covariate_dims,
        color=color,
        shape=shape,
        data=data,
        size=size,
        alpha=alpha,
        figsize=figsize,
    )


def covariates_weight_scatter(
    model: types.MofaFlex | MOFAFLEX,
    factor: int | str,
    groups: str | Sequence[str] | None = None,
    covariate_dims: int | str | Sequence[int] | Sequence[str] | None = None,
    color: int | str | None = None,
    shape: str | None = None,
    data: MuData | Mapping[str, Mapping[str, AnnData]] | AnnData | None = None,
    size: float = 1,
    alpha: float = 1,
    figsize: tuple[float, float] = (6, 6),
) -> p9.ggplot:
    """Plot a factor against one or two covariate dimensions.

    Args:
        model: The term to plot the factor correlation for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        factor: The factor to plot.
        groups: The groups to plot. If `None`, all groups with covariates are shown.
        covariate_dims: The dimensions of the covariates to plot against. If a list of length 1, plot covariate
            on the x-axis and factor on the y-axis. If a list of length 2, plot the first covariate on the x-axis,
            the second covariate on the y-axis, and factor as color. If None, use all dimensions.
        color: The factor or covariate to color by. Only used when one covariate dimension is plotted.
        shape: The covariate name to shape by.
        data: The data that the model was trained on. Only required if `color is not None` or `shape is not None`.
        size: Size of the data points.
        alpha: Transparency of the data points.
        figsize: Figure size in inches.
    """
    return _plot_covariates_scatter(
        model,
        axis=1,
        factor=factor,
        groups=groups,
        covariate_dims=covariate_dims,
        color=color,
        shape=shape,
        data=data,
        size=size,
        alpha=alpha,
        figsize=figsize,
    )


def training_curve(
    model: MOFAFLEX, linecolor: str = "#214D83", linewidth: int = 1, figsize: tuple[float, float] = (12, 4)
) -> p9.ggplot:
    """Plot the training curve: -ELBO vs epoch.

    Args:
        model: The MOFA-FLEX model to plot the training curve for.
        linecolor: The color of the line.
        linewidth: The width of the line.
        figsize: Figure size in inches.
    """
    df = pd.DataFrame({"Epoch": range(len(model.training_loss)), "-ELBO": model.training_loss})

    plot = (
        p9.ggplot(df, p9.aes(x="Epoch", y="-ELBO"))
        + p9.geom_line(color=linecolor, size=linewidth)
        + p9.labs(title="Training Curve", x="Epoch", y="-ELBO")
        + p9.theme(figure_size=figsize)
    )

    return plot


def factor_correlation(model: types.MofaFlex | MOFAFLEX, figsize: tuple[float, float] = (8, 8)) -> p9.ggplot:
    """Plot the correlation between factors.

    Args:
        model: The term to plot the factor correlation for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        figsize: Figure size in inches.
    """
    all_corr_dfs = []

    for group_name, corr_df in tl.factor_correlation(model).items():
        corr_df.reset_index(names="factor1", inplace=True)
        corr_df = corr_df.melt("factor1", var_name="factor2", value_name="correlation")
        corr_df["group"] = group_name
        all_corr_dfs.append(corr_df)

    final_df = pd.concat(all_corr_dfs, axis=0, ignore_index=True).assign(
        factor1=lambda x: pd.Categorical(x.factor1, categories=x.factor1.unique()),
        factor2=lambda x: pd.Categorical(x.factor2, categories=x.factor2.unique()),
    )

    plot = (
        p9.ggplot(final_df, p9.aes(x="factor1", y="factor2", fill="correlation"))
        + p9.geom_tile()
        + _scale_fill_zerosymmetric_diverging(limits=(-1, 1), expand=(0, 0), name="Correlation")
        + p9.scale_x_discrete(expand=(0, 0))
        + p9.scale_y_discrete(expand=(0, 0))
        + p9.coord_equal()
        + p9.labs(x="", y="")
        + p9.theme(
            figure_size=figsize,
            axis_text_x=p9.element_text(angle=45, hjust=1),
            panel_grid_major=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
            **_no_axis_ticks_x,
            **_no_axis_ticks_y,
        )
        + p9.facet_wrap("group")
    )

    return plot


def overview(
    data: MuData | Mapping[str, Mapping[str, AnnData]] | AnnData | MofaFlexDataset,
    group_by: str | Sequence[str] | None = None,
    missingcolor: str = "#214D83",
    nonmissingcolor: str = "#8AB6D4",
    figsize: tuple[float, float] = (15, 5),
    max_plot_obs: int = 400,
    max_plot_x_labels: int = 150,
) -> p9.ggplot:
    """Generate an overview plot of missing data across different views and groups.

    Args:
        data: A nested dictionary where the first level keys are group names,
            and the second level keys are view names. The values are AnnData objects.
        group_by: Columns of `.obs` in :class:`MuData<mudata.MuData>` or :class:`AnnData<anndata.AnnData>` objects to group data by.
            Ignored if the input data is not a :class:`MuData<mudata.MuData>` or :class:`AnnData<anndata.AnnData>` object.
        missingcolor: The color to use for missing data.
        nonmissingcolor: The color to use for non-missing data.
        figsize: Figure size in inches.
        max_plot_obs: The maximum number of observations to plot. If the number of observations is greater than this value in any group,
            a horizontal bar plot is created instead of a tile plot.
        max_plot_x_labels: The maximum number of x-axis labels to show. If the number of observations is greater than this value in any group,
            the x-axis labels are not shown.
    """
    from .._core import MofaFlexDataset

    if not isinstance(data, MofaFlexDataset):
        data = MofaFlexDataset(data, group_by=group_by)

    missings = data.get_missing_obs()
    n_obs_groups = max(data.n_samples.values())

    # if the number of observations in every group is low, plot every observation
    if n_obs_groups < max_plot_obs:
        theme = p9.theme(figure_size=figsize, **_no_axis_ticks_y)
        if n_obs_groups < max_plot_x_labels:
            theme += p9.theme(axis_text_x=p9.element_text(angle=90, ha="center", va="top"))
        else:
            theme += p9.theme(axis_text_x=p9.element_blank(), axis_ticks_x=p9.element_blank())

        plot = (
            p9.ggplot(missings, p9.aes(x="obs_name", y="view", fill="missing"))
            + p9.geom_tile()
            + p9.facet_wrap("group", scales="free_x")
            + p9.scale_fill_manual(values=[missingcolor, nonmissingcolor])
            + p9.scale_x_discrete(expand=(0, 0))
            + p9.scale_y_discrete(expand=(0, 0))
            + theme
            + p9.labs(x="Observations", y="")
        )

    # otherwise, make a barplot showing the number of observations
    else:
        obs_counts = missings[~missings.missing].groupby(["group", "view"]).size().reset_index(name="count")

        plot = (
            p9.ggplot(obs_counts, p9.aes(x="view", y="count"))
            + p9.geom_bar(stat="identity", fill=missingcolor)
            + p9.facet_wrap("group", scales="free_x")
            + p9.scale_y_continuous(expand=(0, 0.0, 0.05, 0))
            + p9.theme(figure_size=figsize, **_no_axis_ticks_y)
            + p9.labs(y="Number of non-missing observations", x="")
            + p9.coord_flip()
        )

    return plot


def variance_explained(
    model: MOFAFLEX,
    group_by: Literal["group", "view"] = "group",
    term: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot the fraction of variance explained per factor in each group and view.

    Args:
        model: The MOFA-FLEX model.
        group_by: The grouping to use for the plots.
        term: The name of the additive term to plot the variance explained for. If `None` and the model has only one additive term,
            will plot the fraction of variance explained per factor for this term. If `None` and the model has multiple terms, will
            plot the fraction of variance explained per term.
        figsize: Figure size in inches.
    """
    if group_by == "group":
        x = "view"
    elif group_by == "view":
        x = "group"
    else:
        raise ValueError("`group_by` argument must be either 'group' or 'view'.")

    if figsize is None:
        figsize = (len(model.group_names) * 3, 5)

    byterm = term is None and model.n_terms > 1
    col = "term" if byterm else "component"
    df_r2 = model.get_r2("byterm" if byterm else "term", ordered=True, term=term).assign(
        factor=lambda x: pd.Categorical(x[col], categories=x[col].unique())
    )
    heatmap = (
        p9.ggplot(df_r2, p9.aes(x=x, y="factor", fill="R2"))
        + p9.geom_tile()
        + p9.scale_fill_distiller(palette="OrRd", limits=(0, None), expand=(0, 0, 1.1, 0), name="Variance\nexplained")
        + p9.scale_x_discrete(expand=(0, 0))
        + p9.scale_y_discrete(expand=(0, 0))
        + p9.labs(x="", y="")
        + p9.theme(
            axis_text_x=p9.element_text(rotation=90), figure_size=figsize, **_no_axis_ticks_x, **_no_axis_ticks_y
        )
        + p9.facet_wrap(group_by)
    )

    return heatmap


def factor_significance(
    model: types.MofaFlex | MOFAFLEX,
    n_factors: int | None = None,
    views: str | Sequence[str] | None = None,
    groups: str | Sequence[str] | None = None,
    alpha: float = 0.05,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot an overview of the factors summarizing the PCGSE results along with the variance explained per factor.

    This is a diagnostic plot showing only the results of testing a factor against its matching annotation.  Of the
    two one-sided tests, only the most significant one is shown. The factor names of the factors significant at
    `alpha` FDR will be annotated with the direction of the test.

    Args:
        model: The term to plot the factor significance for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        n_factors: Number of top factors to plot. If `None`, plot all factors (ordered).
        views: The views to consider in the ranking. If `None`, plot all views.
        groups: The groups to consider in the ranking. If `None`, plot all groups.
        alpha: False discovery rate threshold.
        figsize: Figure size in inches.
    """
    pcgse_results = model.get_significant_annotations()
    if pcgse_results is None:
        raise ValueError("PCGSE results not available.")

    if views is None:
        # TODO: I'd prefer we plot even without pcgse results, simply to show variance explained (?)
        views = [view for view in model.view_names if view in pcgse_results]
    elif isinstance(views, str):
        views = [views]

    if groups is None:
        groups = model.group_names
    elif isinstance(groups, str):
        groups = [groups]

    if figsize is None:
        figsize = (6 * len(views), 4 * len(groups))

    pcgse_results = (
        pd.concat(
            {
                view_name: pcgse_results[view_name].loc[
                    (pcgse_results[view_name]["factor"] == pcgse_results[view_name]["annotation"]), :
                ]
                for view_name in views
            },
            axis=0,
        )
        .reset_index(level=0, names="view")
        .reset_index(drop=True)
    )
    r2 = model.get_r2("term")
    annotations = pd.concat(
        {view_name: vannot.sum(axis=0) for view_name, vannot in model.get_annotations().items()}, axis=0
    ).rename_axis(index=("view", "annotation"))
    factor_order = model.factor_names[model.factor_order]

    combined_df = (
        pcgse_results.loc[pcgse_results.groupby(["view", "annotation"])["padj"].idxmin()]
        .merge(r2, left_on=["view", "factor"], right_on=["view", "component"])
        .set_index(["view", "annotation"])
        .drop(columns="component")
        .assign(
            sign=lambda x: np.where(x["padj"] > alpha, pd.NA, x["sign"]),
            factor=lambda x: x["factor"] + x["sign"].map({"pos": " (+)", "neg": " (-)", pd.NA: ""}),
            annotation_size=annotations,
        )
        .reset_index(level=0)
        .groupby("view")
        .apply(
            lambda df: df.loc[
                factor_order[np.isin(factor_order, df.index)][
                    slice(n_factors) if n_factors is not None else slice(None)
                ],
                :,
            ],
            include_groups=False,
        )
        .assign(factor=lambda x: pd.Categorical(x["factor"], categories=x["factor"].unique()))
        .reset_index()
    )

    combined_scatter = (
        p9.ggplot(combined_df, p9.aes(x="R2", y="factor", fill="-np.log10(padj)", size="annotation_size"))
        + p9.geom_point()
        + p9.scale_fill_distiller(palette="OrRd", limits=(0, 10), oob=bounds.squish)
        + p9.labs(x="$R^2$", y="Factor", fill="$-\\log_{10}(\\text{FDR})$", size="No. features\nin annotation")
        + p9.theme(figure_size=figsize)
    )

    facet = (
        p9.facet_grid("group", "view")
        if len(groups) > 1 and len(views) > 1
        else p9.facet_wrap("group")
        if len(groups) > 1
        else p9.facet_wrap("view")
        if len(views) > 1
        else None
    )

    if facet is not None:
        combined_scatter += facet

    return combined_scatter


def all_weights(
    model: types.MofaFlex | MOFAFLEX,
    views: str | Sequence[str] | None = None,
    clip: tuple[float, float] | None = (-1, 1),
    show_featurenames: bool = False,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot the weight matrices.

    Args:
        model: The term to plot the weights for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        views: The views to consider in the ranking. If `None`, plot all views.
        clip: Weight value range to clip to.
        show_featurenames: Whether to show the feature names on the Y axis.
        figsize: Figure size in inches.
    """
    if isinstance(views, str):
        views = [views]
    weights = model.get_weights()
    if figsize is None:
        figsize = (3 * len(weights), 5)
    dfs = []
    for k, df in weights.items():
        if views is None or k in views:
            df.reset_index(names="feature", inplace=True)
            df = df.melt("feature", var_name="factor", value_name="weight")
            df["view"] = k
            dfs.append(df)

    df = pd.concat(dfs, axis=0, ignore_index=True).assign(
        factor=lambda x: pd.Categorical(x.factor, categories=x.factor.unique())
    )
    if clip is not None:
        scale_kwargs = {"oob": bounds.squish, "limits": clip, "expand": (0, 0)}
    else:
        scale_kwargs = {}
    plt = (
        p9.ggplot(df, p9.aes("factor", "feature", fill="weight"))
        + p9.geom_tile()
        + _scale_fill_zerosymmetric_diverging(name="Weight", **scale_kwargs)
        + p9.theme(figure_size=figsize, axis_text_x=p9.element_text(rotation=90))
        + p9.facet_wrap("view", scales="free_y")
        + p9.labs(x="", y="")
    )
    if not show_featurenames:
        plt = plt + p9.theme(axis_text_y=p9.element_blank(), axis_ticks_y=p9.element_blank())
    return plt


def factor(
    model: types.MofaFlex | MOFAFLEX,
    factor: int | str = 1,
    show_samplenames: bool = False,
    figsize: tuple[float, float] | None = None,  # F821
) -> p9.ggplot:
    """Plot factor values (y-axis) for each sample (x-axis).

    Args:
        model: The term to plot the factor values for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        factor: The factor to plot.
        show_samplenames: Whether to show the feature names on the X axis.
        figsize: Figure size in inches.
    """
    factors = model.get_factors(ordered=False)
    if figsize is None:
        figsize = (5, 3 * len(factors))
    if isinstance(factor, str):
        factor = np.nonzero(model.factor_names == factor)[0][0]
    else:
        factor -= 1
    df = []
    for group_name, facs in factors.items():
        df.append(facs.iloc[:, [factor]].reset_index(names="sample").assign(group=group_name))
    df = pd.concat(df, axis=0, ignore_index=True)
    colname = df.columns[1]

    plt = (
        p9.ggplot(df, p9.aes("sample", colname, color=colname))
        + p9.geom_point()
        + p9.geom_hline(yintercept=0, linetype="dotted", color="gray")
        + _scale_color_zerosymmetric_diverging()
        + p9.facet_wrap("group", ncol=1, scales="free_y")
        + p9.theme(
            figure_size=figsize,
            axis_text_x=p9.element_text(rotation=90),
            panel_grid_major_x=p9.element_blank(),
            panel_grid_minor_x=p9.element_blank(),
        )
    )
    if not show_samplenames:
        plt = plt + p9.theme(axis_text_x=p9.element_blank(), axis_ticks_x=p9.element_blank())
    return plt


def _check_covariate(cov, group_name, covars):
    if isinstance(cov, str):
        covidx = np.nonzero(covars.columns == cov)[0]
        if not covidx.size:
            raise ValueError(f"Covariate `{cov}` not found in group `{group_name}`.")
        elif covidx.size > 1:
            raise ValueError(f"Multiple covariates named `{cov}` found in group `{group_name}`.")
        else:
            cov = covidx[0]
    elif cov < 0 or cov >= covars.shape[1]:
        raise ValueError(f"Group {group_name} has {covars.shape[1]} covariates, but covariate {cov} requested.")
    return cov


def _plot_covariate(
    covariates: Mapping[str, pd.DataFrame],
    factors: Mapping[str, pd.DataFrame],
    n_factors: int,
    covariate1: str | int,
    covariate2: str | int | None = None,
    size: int = 1,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot every factor against one or two covariates.

    Args:
        covariates: The covariates dict from the model.
        factors: The factors dict from the model.
        n_factors: The number of factors.
        covariate1: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
        covariate2: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
            If `None`, only one covariate will be plotted.
        size: The point size.
        figsize: Figure size in inches.
    """
    if figsize is None:
        figsize = (2 * n_factors, 2 * len(factors))

    df = []
    covnames = [covariate1, covariate2]
    for i, cov in enumerate((covariate1, covariate2)):
        if not isinstance(cov, str):
            covnames[i] = f"Covariate {cov}"

    if covariate2 is not None:
        covcheck = (covariate1, covariate2)
        id_vars = ["cov1", "cov2"]
        yaes = "cov2"
        ylab = covnames[1]
    else:
        covcheck = (covariate1,)
        id_vars = ["cov1"]
        yaes = "value"
        ylab = "Factor value"

    for group_name, covars in covariates.items():
        covidxs = tuple(_check_covariate(cov, group_name, covars) for cov in covcheck)

        cdf = factors[group_name].assign(cov1=covars.iloc[:, covidxs[0]], group=group_name)
        if len(covidxs) > 1:
            cdf = cdf.assign(cov2=covars.iloc[:, covidxs[1]])
        df.append(cdf.reset_index(names="sample"))

    df = (
        pd.concat(df, axis=0, ignore_index=True)
        .melt(id_vars=["group", "sample"] + id_vars, var_name="factor", value_name="value")
        .assign(factor=lambda x: pd.Categorical(x.factor, categories=x.factor.unique()))
    )

    plt = (
        p9.ggplot(df, p9.aes("cov1", yaes, color="value"))
        + p9.geom_point(size=size)
        + _scale_color_zerosymmetric_diverging()
        + p9.labs(x=covnames[0], y=ylab, color="Factor value")
        + p9.facet_grid("group", "factor")
        + p9.theme(figure_size=figsize)
    )
    if covariate2 is None:
        plt = plt + p9.geom_hline(yintercept=0, linetype="dotted", color="gray")
    return plt


def factors_covariate(
    model: types.MofaFlex | MOFAFLEX,
    covariate1: str | int,
    covariate2: str | int | None = None,
    size: int = 1,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot every factor against one or two covariates.

    Args:
        model: The term to plot the factors for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        covariate1: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
        covariate2: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
            If `None`, only one covariate will be plotted.
        size: The point size.
        figsize: Figure size in inches.
    """
    return _plot_covariate(
        model.factor_covariates,
        model.get_factors(),
        model.n_total_factors,
        covariate1,
        covariate2,
        size=size,
        figsize=figsize,
    )


def weights_covariate(
    model: types.MofaFlex | MOFAFLEX,
    covariate1: str | int,
    covariate2: str | int | None = None,
    size: int = 1,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot the weights of every factor against one or two covariates.

    Args:
        model: The term to plot the factors for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        covariate1: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
        covariate2: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
            If `None`, only one covariate will be plotted.
        size: The point size.
        figsize: Figure size in inches.
    """
    return _plot_covariate(
        model.weight_covariates,
        model.get_weights(),
        model.n_total_factors,
        covariate1,
        covariate2,
        size=size,
        figsize=figsize,
    )


def _plot_gp_covariate(
    covariates: Mapping[str, pd.DataFrame],
    gp_means: Mapping[str, pd.DataFrame],
    gp_stds: Mapping[str, pd.DataFrame],
    n_total_factors: int,
    ci_opacity: float = 0.3,
    group: Literal["facet", "color"] = "facet",
    color: str = "black",
    size: int = 1,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    covdim = np.unique(tuple(cov.shape[1] for cov in covariates.values()))
    if covdim.size > 1:
        raise NotImplementedError("Different groups have different covariate dimensions.")
    elif covdim[0] > 2:
        raise NotImplementedError("Cannot plot covariates with >2 dimensions.")

    covnames = [np.asarray(tuple(covar.columns[i] for covar in covariates.values())) for i in range(covdim[0])]
    for i in range(covdim[0]):
        if covnames[i].size == 0:
            covnames[i] = f"Covariate {i}"
        else:
            covname = np.unique(covnames[i])
            if covname.size > 1:
                covnames[i] = f"Covariate {i}"
            else:
                covnames[i] = covname[0]
    if covdim[0] == 2:
        return _plot_covariate(covariates, gp_means, n_total_factors, 0, 1, size=size, figsize=figsize) + p9.labs(
            x=covnames[0], y=covnames[1]
        )

    if figsize is None:
        figsize = (2 * n_total_factors, 2 * len(gp_means))

    df = []

    for group_name, covars in covariates.items():
        cdf_mean = (
            gp_means[group_name]
            .assign(cov=covars.iloc[:, 0])
            .reset_index(names="sample")
            .melt(id_vars=["sample", "cov"], var_name="factor", value_name="mean")
        )
        cdf_std = (
            gp_stds[group_name].reset_index(names="sample").melt(id_vars="sample", var_name="factor", value_name="std")
        )
        df.append(pd.merge(cdf_mean, cdf_std, on=["sample", "factor"]).assign(group=group_name))

    df = pd.concat(df, axis=0, ignore_index=True).assign(
        factor=lambda x: pd.Categorical(x.factor, categories=x.factor.unique())
    )

    aes_kwargs = {}
    ribbon_kwargs = {}
    line_kwargs = {}
    if group == "facet":
        ribbon_kwargs["fill"] = color
        line_kwargs["color"] = color
    else:
        aes_kwargs["fill"] = aes_kwargs["color"] = "group"
    plt = (
        p9.ggplot(df, p9.aes("cov", "mean", ymax="mean + 1.96 * std", ymin="mean - 1.96 * std", **aes_kwargs))
        + p9.geom_ribbon(alpha=ci_opacity, size=0, **ribbon_kwargs)
        + p9.geom_line(**line_kwargs)
        + p9.labs(x=covnames[0], y="Factor value")
        + p9.theme(figure_size=figsize)
    )
    if group == "facet":
        plt = plt + p9.facet_grid("group", "factor")
    else:
        plt = plt + p9.facet_wrap("factor", nrow=1)
    return plt


def factor_gp_covariates(
    model: types.MofaFlex | MOFAFLEX,
    ci_opacity: float = 0.3,
    group: Literal["facet", "color"] = "facet",
    color: str = "black",
    size: int = 1,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot the fitted GP mean for each factor in each group at the data covariate locations.

    If the model covariates are 2D, plot the covariate on X and Y and encode the GP posterior mean with color.
    If the model covariates are 1D, plot the covariate on X and the GP posterior mean and 95% confidence interval on Y.

    Args:
        model: The term to plot the factors for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        ci_opacity: Opacity of the 95% CI band. Only relevant for 1D covariates.
        group: Whether to encode the sample groups by color or by faceting. Only relevant for 1D covariates.
        color: Color of the line and CI and. Only relevant for 1D covariates and `group="facet"`.
        size: The point size. Only relevant for 2D covariates.
        figsize: Figure size in inches.
    """
    return _plot_gp_covariate(
        model.factor_covariates,
        model.get_factor_gps(),
        model.get_factor_gps(moment="std"),
        model.n_total_factors,
        ci_opacity,
        group,
        color,
        size,
        figsize,
    )


def weight_gp_covariates(
    model: types.MofaFlex | MOFAFLEX,
    ci_opacity: float = 0.3,
    group: Literal["facet", "color"] = "facet",
    color: str = "black",
    size: int = 1,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot the fitted GP mean for each factor in each view at the data covariate locations.

    If the model covariates are 2D, plot the covariate on X and Y and encode the GP posterior mean with color.
    If the model covariates are 1D, plot the covariate on X and the GP posterior mean and 95% confidence interval on Y.

    Args:
        model: The term to plot the factors for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        ci_opacity: Opacity of the 95% CI band. Only relevant for 1D covariates.
        group: Whether to encode the sample groups by color or by faceting. Only relevant for 1D covariates.
        color: Color of the line and CI and. Only relevant for 1D covariates and `group="facet"`.
        size: The point size. Only relevant for 2D covariates.
        figsize: Figure size in inches.
    """
    return _plot_gp_covariate(
        model.weight_covariates,
        model.get_weight_gps(),
        model.get_weight_gps(moment="std"),
        model.n_total_factors,
        ci_opacity,
        group,
        color,
        size,
        figsize,
    )


def _plot_gp_smoothness(
    scale: np.ndarray, factor_names: np.ndarray, figsize: tuple[float, float] = (3, 3)
) -> p9.ggplot:
    if scale is None:
        raise ValueError("model does not have any groups with a GP prior.")
    df = pd.DataFrame({"smoothness": scale, "factor": pd.Categorical(factor_names, categories=factor_names)})
    plt = (
        p9.ggplot(df, p9.aes("factor", "smoothness"))
        + p9.geom_bar(stat="identity")
        + p9.labs(x="", y="Smoothness")
        + p9.scale_y_continuous(limits=(0, 1), expand=(0, 0))
        + p9.theme(figure_size=figsize, axis_text_x=p9.element_text(rotation=90), **_no_axis_ticks_x)
    )
    return plt


def factor_gp_smoothness(model: types.MofaFlex | MOFAFLEX, figsize: tuple[float, float] = (3, 3)) -> p9.gglot:
    """Plot the smoothness of the GP for each factor.

    Args:
        model: The term to plot the smoothness for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        figsize: Figure size in inches.
    """
    return _plot_gp_smoothness(model.factor_gp_scale, model.factor_names, figsize)


def weight_gp_smoothness(model: types.MofaFlex | MOFAFLEX, figsize: tuple[float, float] = (3, 3)) -> p9.gglot:
    """Plot the smoothness of the GP for each factor.

    Args:
        model: The term to plot the smoothness for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        figsize: Figure size in inches.
    """
    return _plot_gp_smoothness(model.weight_gp_scale, model.factor_names, figsize)


def _prepare_weights_df(
    model: types.MofaFlex | MOFAFLEX,
    n_features: int = 10,
    views: str | Sequence[str] | None = None,
    factors: int | str | Sequence[int] | Sequence[str] | None = None,
):
    weights = model.get_weights(ordered=False)
    try:
        annotations = model.get_annotations(ordered=False)
    except AttributeError:
        annotations = {}
    if views is None:
        views = model.view_names
    elif isinstance(views, str):
        views = [views]
    if factors is None:
        factors = np.arange(model.n_total_factors)
    else:
        if not isinstance(factors, Sequence) or isinstance(factors, str):
            factors = (factors,)
        if all(isinstance(factor, str) for factor in factors):
            factors = np.nonzero(np.isin(model.factor_names, factors))[0] + 1
        factors = np.asarray(factors) - 1

    df = []
    have_annot = False

    for view in views:
        cdf = (
            weights[view]
            .iloc[:, factors]
            .reset_index(names="feature")
            .melt(id_vars="feature", var_name="factor", value_name="weight")
        )
        if view in annotations:
            cdf = pd.merge(
                cdf,
                annotations[view]
                .reset_index(names="feature")
                .melt(id_vars="feature", var_name="factor", value_name="annotation"),
                how="left",
                on=["factor", "feature"],
            )
            cdf.fillna({"annotation": False}, inplace=True)
            have_annot = True
        else:
            cdf = cdf.assign(annotation=False)
        df.append(
            cdf.assign(
                view=view,
                weightabs=lambda x: x.weight.abs(),
                factor=lambda x: pd.Categorical(
                    x.factor, categories=model.factor_names
                ).remove_unused_categories(),  # need categorical for proper ordering of factors in the plot, need to remove unused categories due to https://github.com/has2k1/plotnine/issues/930
                inferred=lambda x: ~x.annotation & x.weightabs > 0,
            )
        )

    df = pd.concat(df, axis=0, ignore_index=False)
    return views, factors, df, have_annot


_weights_inferred_color_scale = p9.scale_color_manual(
    values=("red", "black"), breaks=(True, False), labels=("Inferred", "Annotated")
)


def top_weights(
    model: types.MofaFlex | MOFAFLEX,
    n_features: int = 10,
    views: str | Sequence[str] | None = None,
    factors: int | str | Sequence[int] | Sequence[str] | None = None,
    figsize: tuple[int, int] = (5, 5),
    nrow: int | None = None,
    ncol: int | None = None,
) -> p9.ggplot:
    """Plot the top weights for a given factor and view.

    Args:
        model: The term to plot the top weights for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        n_features: Number of top features to plot.
        views: The views to consider in the ranking. If `None`, plot all views.
        factors: The factors to plot. If `None`, plot all factors.
        figsize: Figure size in inches.
        nrow: Number of rows in the faceted plot. If None, plotnine will determine automatically.
        ncol: Number of columns in the faceted plot. If None, plotnine will determine automatically.
    """
    views, factors, df, have_annot = _prepare_weights_df(model, n_features, views, factors)
    df = (
        df.groupby("factor", observed=True)
        .apply(lambda x: x.iloc[x.weightabs.argsort()[-n_features:], :], include_groups=False)
        .reset_index(0)
        .reset_index(drop=True)
        .assign(weightsgn=lambda x: x.weight >= 0)
        .sort_values(["factor", "weightabs"], ascending=True)
    )
    if (
        len(views) > 1
        and (df.groupby("factor", observed=True)["feature"].aggregate(lambda x: x.duplicated().sum()) > 0).any()
    ):
        df = df.assign(feature=lambda x: x.feature.str + "_" + x.view.str)
    df = df.assign(feature=lambda x: pd.Categorical(x.feature, categories=x.feature.unique()))

    aes_kwargs = {}
    if have_annot:
        aes_kwargs["color"] = "inferred"

    plot = (
        p9.ggplot(df, p9.aes("weightabs", "feature", xend=0, yend="feature", shape="weightsgn", **aes_kwargs))
        + p9.geom_segment()
        + p9.geom_point(size=5, stroke=0)
        + p9.scale_shape_manual(values=("$\\oplus$", "$\\ominus$"), breaks=(True, False), guide=None)
        + _weights_inferred_color_scale
        + p9.scale_x_continuous(expand=(0, 0, 0.05, 0))
        + p9.labs(x="| Weight |", y="", color="")
        + p9.theme(figure_size=figsize, **_no_axis_ticks_y)
    )

    plot += p9.facet_wrap("factor", scales="free", nrow=nrow, ncol=ncol)

    return plot


def weights(
    model: types.MofaFlex | MOFAFLEX,
    n_features: int = 10,
    views: str | Sequence[str] | None = None,
    factors: int | str | Sequence[int] | Sequence[str] | None = None,
    pointsize: float = 2,
    figsize: tuple[int, int] | None = None,
    nrow: int | None = None,
    ncol: int | None = None,
) -> p9.ggplot:
    """Plot the weights for a given factor and view.

    Args:
        model: The term to plot the weights for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        n_features: Number of top features to annotate.
        views: The views to consider in the ranking. If `None`, use all views.
        factors: The factors to plot. If `None`, plot all factors.
        pointsize: Point size for the annotated features. Points for unannotated features will be
            of size `0.25 * pointsize`.
        figsize: Figure size in inches.
        nrow: Number of rows in the faceted plot. If None, plotnine will determine automatically.
        ncol: Number of columns in the faceted plot. If None, plotnine will determine automatically.
    """
    views, factors, df, have_annot = _prepare_weights_df(model, n_features, views, factors)
    if figsize is None:
        figsize = (3 * len(factors), 3 * len(views))
        if p9.options.limitsize:
            figsize = (min(figsize[0], 25), min(figsize[1], 25))

    grp = df.groupby(["factor", "view"], observed=True)
    df["rank"] = grp["weight"].rank(ascending=False, method="min")
    df["absrank"] = grp["weightabs"].rank(ascending=False, method="min")
    df["annotate"] = df["absrank"] <= n_features

    aes_kwargs = {}
    if have_annot:
        aes_kwargs["color"] = "inferred"

    labeled_data = df[df.annotate].assign(ha=lambda x: np.where(x["weight"] > 0, "left", "right"))
    y_max = labeled_data.groupby("view", observed=True)["weight"].max()
    y_min = labeled_data.groupby("view", observed=True)["weight"].min()

    labeled_groups = []
    for (view, _), cdf in labeled_data.groupby(["view", "factor"], observed=True):
        n_positive = (cdf["weight"] > 0).sum()
        n_negative = n_features - n_positive

        cdf.sort_values("rank", inplace=True)
        cdf["y_text_pos"] = np.concatenate(
            [
                np.linspace(y_max[view], 0.1 * y_max[view], num=n_positive),
                np.linspace(0.1 * y_min[view], y_min[view], num=n_negative),
            ]
        )
        labeled_groups.append(cdf)
    labeled_data = pd.concat(labeled_groups, axis=0, ignore_index=True)

    labeled_data["x_text_pos"] = df["rank"].max() / 2

    plot = (
        p9.ggplot(df, p9.aes("rank", "weight", label="feature", **aes_kwargs))
        + p9.geom_point(p9.aes(size="annotate"), stroke=0)
        + p9.scale_size_manual(breaks=(True, False), values=(pointsize, 0.25 * pointsize), guide=None)
        + _weights_inferred_color_scale
        + p9.labs(x="Rank", y="Weight", color="")
        + p9.theme(figure_size=figsize)
        + p9.geom_text(
            data=labeled_data,
            mapping=p9.aes(x="x_text_pos", y="y_text_pos", label="feature", ha="ha"),
            size=10,
            va="center",
            show_legend=False,
        )
        + p9.geom_segment(
            data=labeled_data, mapping=p9.aes(x="rank", y="weight", xend="x_text_pos", yend="y_text_pos"), color="gray"
        )
    )

    if nrow is not None or ncol is not None:
        plot += p9.facet_wrap(["view", "factor"], scales="free_y", nrow=nrow, ncol=ncol)
    else:
        plot += p9.facet_grid("view", "factor", scales="free_y")

    return plot


def _plot_sparse_probabilities_histogram(
    probs: dict[str, pd.DataFrame], bins: int = 50, nrow: int | None = None, ncol: int | None = None
):
    df = []
    for name, cdf in probs.items():
        df.append(pd.DataFrame({"key": name, "prob": cdf.to_numpy().reshape(-1)}))
    df = pd.concat(df, axis=0, ignore_index=True)
    plot = (
        p9.ggplot(df, p9.aes("prob"))
        + p9.geom_histogram(bins=bins)
        + p9.facet_wrap("key", scales="free", nrow=nrow, ncol=ncol)
        + p9.labs(x="Probability")
        + p9.scale_x_continuous(expand=(0, 0))
        + p9.scale_y_continuous(expand=(0, 0, 0.05, 0))
    )
    return plot


def weight_sparsity_histogram(
    model: types.MofaFlex | MOFAFLEX, bins: int = 50, nrow: int | None = None, ncol: int | None = None
):
    """Plot a histogram of probabilities that weights are non-zero for views with SnS prior.

    The spike-and-slab prior is a mixture distribution of a Normal distribution with a
    potentially non-zero mean and a Dirac delta distribution at zero. This function plots
    the posterior probability of the Normal mixture compoonent.

    Args:
        model: The term to plot the histogram for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        bins: Number of histogram bins.
        nrow: Number of rows in the faceted plot. If None, plotnine will determine automatically.
        ncol: Number of columns in the faceted plot. If None, plotnine will determine automatically.
    """
    return _plot_sparse_probabilities_histogram(
        model.get_sparse_weight_probabilities(), bins=bins, nrow=nrow, ncol=ncol
    )


def factor_sparsity_histogram(
    model: types.Mofaflex | MOFAFLEX, bins: int = 50, nrow: int | None = None, ncol: int | None = None
):
    """Plot a histogram of probabilities that factors are non-zero for views with SnS prior.

    The spike-and-slab prior is a mixture distribution of a Normal distribution with a
    potentially non-zero mean and a Dirac delta distribution at zero. This function plots
    the posterior probability of the Normal mixture compoonent.

    Args:
        model: The term to plot the histogram for. Can also be a :class:`~mofaflex.MOFAFLEX` object if it has only one term.
        bins: Number of histogram bins.
        nrow: Number of rows in the faceted plot. If None, plotnine will determine automatically.
        ncol: Number of columns in the faceted plot. If None, plotnine will determine automatically.
    """
    return _plot_sparse_probabilities_histogram(
        model.get_sparse_factor_probabilities(), bins=bins, nrow=nrow, ncol=ncol
    )
