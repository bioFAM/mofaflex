from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import plotnine as p9
from anndata import AnnData
from mizani import bounds
from mizani.palettes import brewer_pal
from mudata import MuData

if TYPE_CHECKING:
    from .._core import PRISMO, PrismoDataset


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


def plot_factors_scatter(
    model: PRISMO,
    x: int,
    y: int,
    group: str | None = None,
    color: str | None = None,
    shape: str | None = None,
    size: float = 2,
    alpha: float = 1,
    figsize: tuple[float, float] = (6, 6),
) -> p9.ggplot:
    """Plot two factors against each other and color by covariates.

    Args:
        model: A PRISMO model.
        x: The factor to plot on the x-axis.
        y: The factor to plot on the y-axis.
        group: The name of the group. If None, we use the first group.
        color: The covariate name to color by.
        shape: The covariate name to shape by.
        size: Size of the data points.
        alpha: Transparency of the data points.
        figsize: Figure size in inches.
    """
    if group is None:
        group = model.group_names[0]

    if x < 0 or y < 0:
        raise ValueError("Factors x and y must be non-negative.")
    if x >= model.n_factors or y >= model.n_factors:
        raise ValueError("Factors x and y must be in range of the number of factors.")

    facs = model.get_factors(return_type="anndata")[group]
    df_factors = pd.concat([facs.to_df(), facs.obs], axis=1)

    if color is not None and color not in df_factors.columns:
        raise ValueError(f"Color variable {color} not found in the data.")
    if shape is not None and shape not in df_factors.columns:
        raise ValueError(f"Shape variable {shape} not found in the data.")

    aes_kwargs = {}
    if color is not None:
        aes_kwargs["color"] = color
    if shape is not None:
        aes_kwargs["shape"] = shape

    plot = (
        p9.ggplot(df_factors, p9.aes(x=f"Factor {x}", y=f"Factor {y}", **aes_kwargs))
        + p9.geom_hline(yintercept=0, linetype="dashed", color="black")
        + p9.geom_vline(xintercept=0, linetype="dashed", color="black")
        + p9.geom_point(size=size, alpha=alpha, stroke=0)
        + p9.theme(figure_size=figsize)
    )

    return plot


def plot_covariates_factor_scatter(
    model: PRISMO,
    factor: int,
    group: str | None = None,
    covariate_dims: list[int] | None = None,
    color: str | None = None,
    shape: str | None = None,
    figsize: tuple[float, float] = (6, 6),
) -> p9.ggplot:
    """Plot a factor against one or two covariate dimensions.

    Args:
        model: A PRISMO model.
        factor: The factor to plot.
        group: The name of the group. If None, we use the first group with a covariate.
        covariate_dims: The dimensions of the covariates to plot against. If a list of length 1, plot covariate
            on the x-axis and factor on the y-axis. If a list of length 2, plot the first covariate on the x-axis,
            the second covariate on the y-axis, and factor as color. If None, use all dimensions.
        color: The covariate name to color by. Only used when one covariate dimension is plotted.
        shape: The covariate name to shape by.
        figsize: Figure size in inches.
    """
    if len(covariate_dims) == 2 and color is not None:
        raise ValueError("Cannot specify a color variable when plotting two covariate dimensions.")
    if factor not in range(1, model.n_factors + 1):
        raise ValueError(f"Factors must be between 1 and {model.n_factors}.")
    if len(covariate_dims) not in (1, 2):
        raise ValueError("Covariate dimensions must be a list of length 1 or 2.")

    if group is None:
        for group_name in model.group_names:
            if group_name in model.covariates:
                group = group_name
                break

    # create factor DataFrame
    facs = model.get_factors(return_type="anndata")[group]
    df_factors = pd.concat([facs.to_df(), facs.obs], axis=1)

    if color is not None and color not in df_factors.columns:
        raise ValueError(f"Color variable {color} not found in the data.")
    if shape is not None and shape not in df_factors.columns:
        raise ValueError(f"Shape variable {shape} not found in the data.")

    covnames = (
        model.covariates_names[group]
        if group in model.covariates_names
        else [f"Covariate {i}" for i in range(model.covariates[group].shape[1])]
    )

    # create covariate DataFrame
    df_covariates = pd.DataFrame(model.covariates[group], index=df_factors.index, columns=covnames)

    # combine factor and covariate DataFrames
    df = pd.concat([df_factors, df_covariates], axis=1)

    aes_kwargs = {}
    if len(covariate_dims) == 1:
        x = df_covariates.columns[covariate_dims[0]]
        y = f"Factor {factor}"
        if color is not None:
            aes_kwargs["color"] = color

    elif len(covariate_dims) == 2:
        x = df_covariates.columns[covariate_dims[0]]
        y = df_covariates.columns[covariate_dims[1]]
        aes_kwargs["color"] = f"Factor {factor}"

    if shape is not None:
        aes_kwargs["shape"] = shape

    plot = p9.ggplot(df, p9.aes(x=x, y=y, **aes_kwargs)) + p9.geom_point(size=0.1) + p9.theme(figure_size=figsize)

    return plot


def plot_training_curve(
    model: PRISMO, linecolor: str = "#214D83", linewidth: int = 1, figsize: tuple[float, float] = (12, 4)
) -> p9.ggplot:
    """Plot the training curve: -ELBO vs epoch.

    Args:
        model: The PRISMO model to plot the training curve for.
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


def plot_factor_correlation(model: PRISMO, figsize: tuple[float, float] = (8, 8)) -> p9.ggplot:
    """Plot the correlation between factors.

    Args:
        model: The model to plot the factor correlation for.
        figsize: Figure size in inches.
    """
    factors = model.get_factors()
    all_corr_dfs = []

    for k, v in factors.items():
        corr_df = pd.DataFrame(np.corrcoef(v.to_numpy().T), index=model.factor_names, columns=model.factor_names)

        corr_df["factor1"] = model.factor_names
        corr_df = corr_df.melt("factor1", var_name="factor2", value_name="correlation")
        corr_df["group"] = k
        all_corr_dfs.append(corr_df)

    final_df = pd.concat(all_corr_dfs, axis=0, ignore_index=True).assign(
        factor1=lambda x: pd.Categorical(x.factor1, categories=x.factor1.unique()),
        factor2=lambda x: pd.Categorical(x.factor2, categories=x.factor2.unique()),
    )

    plot = (
        p9.ggplot(final_df, p9.aes(x="factor1", y="factor2", fill="correlation"))
        + p9.geom_tile()
        + _scale_fill_zerosymmetric_diverging(limits=(-1, 1), expand=(0, 0), name="Correlation")
        + p9.coord_equal()
        + p9.labs(x="", y="")
        + p9.theme(
            figure_size=figsize,
            axis_text_x=p9.element_text(angle=45, hjust=1),
            panel_grid_major=p9.element_blank(),
            panel_grid_minor=p9.element_blank(),
        )
        + p9.facet_wrap("group")
    )

    return plot


def plot_overview(
    data: dict[str, dict[str, AnnData]] | MuData | PrismoDataset,
    group_by: str | list[str] | None = None,
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
        group_by: Columns of `.obs` in :external:py:class:`MuData<mudata.MuData>` objects to group data by.
            Ignored if the input data is not a :external:py:class:`MuData<mudata.MuData>` object.
        missingcolor: The color to use for missing data.
        nonmissingcolor: The color to use for non-missing data.
        figsize: Figure size in inches.
        max_plot_obs: The maximum number of observations to plot. If the number of observations is greater than this value in any group,
            a horizontal bar plot is created instead of a tile plot.
        max_plot_x_labels: The maximum number of x-axis labels to show. If the number of observations is greater than this value in any group,
            the x-axis labels are not shown.
    """
    from .._core import PrismoDataset

    if not isinstance(data, PrismoDataset):
        data = PrismoDataset(data, group_by=group_by)

    missings = data.get_missing_obs()
    n_obs_groups = max(data.n_samples.values())

    # if the number of observations in every group is low, plot every observation
    if n_obs_groups < max_plot_obs:
        plot_x_labels = n_obs_groups < max_plot_x_labels

        plot = (
            p9.ggplot(missings, p9.aes(x="obs_name", y="view", fill="missing"))
            + p9.geom_tile()
            + p9.facet_wrap("group", scales="free_x")
            + p9.scale_fill_manual(values=[missingcolor, nonmissingcolor])
            + p9.theme(
                axis_text_x=(p9.element_text(angle=90, ha="center", va="top") if plot_x_labels else p9.element_blank()),
                figure_size=figsize,
            )
            + p9.labs(title="Data Overview", x="Observation", y="View")
        )

    # otherwise, make a barplot showing the number of observations
    else:
        obs_counts = missings[~missings.missing].groupby(["group", "view"]).size().reset_index(name="count")

        plot = (
            p9.ggplot(obs_counts, p9.aes(x="view", y="count"))
            + p9.geom_bar(stat="identity", fill=missingcolor)
            + p9.facet_wrap("group", scales="free_x")
            + p9.theme(axis_text_x=(p9.element_text(angle=90, ha="center", va="top")), figure_size=figsize)
            + p9.labs(title="Data Overview", y="Number of non-missing observations", x="View")
            + p9.coord_flip()
        )

    return plot


def plot_variance_explained(
    model: PRISMO, group_by: Literal["group", "view"] = "group", figsize: tuple[float, float] | None = None
) -> p9.ggplot:
    """Plot the variance explained per factor in each group and view.

    Args:
        model: The PRISMO model.
        group_by: The grouping to use for the plots.
        figsize: Figure size in inches.
    """
    if group_by == "group":
        x = "view"
    elif group_by == "view":
        x = "group"
    else:
        raise ValueError("`group_by` argument must be either 'group' or 'view'.")

    df_r2 = model.get_r2()

    combined_df = []
    if figsize is None:
        figsize = (len(model.group_names) * 3, 5)

    for group_name, df in df_r2.items():
        r2_df = df.reset_index(names="factor").melt("factor", var_name="view", value_name="var_exp")
        r2_df["group"] = group_name

        combined_df.append(r2_df)
    combined_df = pd.concat(combined_df, ignore_index=True).assign(
        factor=lambda x: pd.Categorical(x.factor, categories=x.factor.unique())
    )

    combined_heatmap = (
        p9.ggplot(combined_df, p9.aes(x=x, y="factor", fill="var_exp"))
        + p9.geom_tile()
        + p9.scale_fill_distiller(palette="OrRd", limits=(0, None), expand=(0, 0, 1.1, 0), name="Variance\nexplained")
        + p9.labs(x=x.capitalize())
        + p9.theme(axis_text_x=p9.element_text(rotation=90), figure_size=figsize)
        + p9.facet_wrap(group_by)
    )

    return combined_heatmap


def plot_all_weights(
    model: PRISMO,
    clip: tuple[float, float] | None = (-1, 1),
    show_featurenames: bool = False,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot the weight matrices.

    Args:
        model: The PRISMO model.
        clip: Weight value range to clip to.
        show_featurenames: Whether to show the feature names on the Y axis.
        figsize: Figure size in inches.
    """
    weights = model.get_weights()
    if figsize is None:
        figsize = (3 * len(weights), 5)
    dfs = []
    for k, df in weights.items():
        df.reset_index(names="factor", inplace=True)
        df = df.melt("factor", var_name="feature", value_name="weight")
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
        + p9.facet_wrap("view")
        + p9.labs(x="", y="")
    )
    if not show_featurenames:
        plt = plt + p9.theme(axis_text_y=p9.element_blank(), axis_ticks_y=p9.element_blank())
    return plt


def plot_factor(
    model: PRISMO,
    factor: int = 1,
    show_featurenames: bool = False,
    figsize: tuple[float, float] | None = None,  # F821
) -> p9.ggplot:
    """Plot factor values (y-axis) for each sample (x-axis).

    Args:
        model: The PRISMO model.
        factor: The factor to plot.
        show_featurenames: Whether to show the feature names on the X axis.
        figsize: Figure size in inches.
    """
    factors = model.get_factors(ordered=False)
    if figsize is None:
        figsize = (5, 3 * len(factors))
    df = []
    for group_name, facs in factors.items():
        df.append(facs.iloc[:, [factor - 1]].reset_index(names="sample").assign(group=group_name))
    df = pd.concat(df, axis=0, ignore_index=True)
    colname = df.columns[1]

    plt = (
        p9.ggplot(df, p9.aes("sample", colname, color=colname))
        + p9.geom_point()
        + p9.geom_hline(yintercept=0, linetype="dotted", color="gray")
        + _scale_color_zerosymmetric_diverging()
        + p9.facet_wrap("group", ncol=1, scales="free_y")
        + p9.theme(figure_size=figsize, axis_text_x=p9.element_text(rotation=90))
    )
    if not show_featurenames:
        plt = plt + p9.theme(axis_text_x=p9.element_blank(), axis_ticks_x=p9.element_blank())
    return plt


def _check_covariate(cov, cnames, group_name, covars):
    if isinstance(cov, str):
        covidx = np.nonzero(cnames == cov)[0]
        if not covidx.size:
            raise ValueError(f"Covariate `{cov}` not found in group `{group_name}`.")
        elif covidx.size > 1:
            raise ValueError(f"Multiple covariates named `{cov}` found in group `{group_name}`.")
        else:
            cov = covidx[0]
    elif cov < 0 or cov >= covars.shape[1]:
        raise ValueError(f"Group {group_name} has {covars.shape[1]} covariates, but covariate {cov} requested.")
    return cov


def _plot_factors_covariate(
    model: PRISMO,
    covariate1: str | int,
    covariate2: str | int | None = None,
    gp: bool = False,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot every factor against one or two covariates.

    Args:
        model: The PRISMO model.
        covariate1: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
        covariate2: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
            If `None`, only one covariate will be plotted.
        gp: If `False`, plot the estimated factor values. If `True`, plot the GP predictions.
        figsize: Figure size in inches.
    """
    factors = model.get_factors() if not gp else model.get_gps()

    if figsize is None:
        figsize = (2 * model.n_factors, 2 * len(factors))

    covariates = model.covariates
    covariate_names = model.covariates_names

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
        cnames = covariate_names.get(group_name, ())
        covidxs = tuple(_check_covariate(cov, cnames, group_name, covars) for cov in covcheck)

        cdf = factors[group_name].reset_index(names="sample").assign(cov1=covars[:, covidxs[0]], group=group_name)
        if len(covidxs) > 1:
            cdf = cdf.assign(cov2=covars[:, covidxs[1]])
        df.append(cdf)

    df = (
        pd.concat(df, axis=0, ignore_index=True)
        .melt(id_vars=["group", "sample"] + id_vars, var_name="factor", value_name="value")
        .assign(factor=lambda x: pd.Categorical(x.factor, categories=x.factor.unique()))
    )

    plt = (
        p9.ggplot(df, p9.aes("cov1", yaes, color="value"))
        + p9.geom_point()
        + _scale_color_zerosymmetric_diverging()
        + p9.labs(x=covnames[0], y=ylab)
        + p9.facet_grid("group", "factor")
        + p9.theme(figure_size=figsize)
    )
    if covariate2 is None:
        plt = plt + p9.geom_hline(yintercept=0, linetype="dotted", color="gray")
    return plt


def plot_factors_covariate(
    model: PRISMO,
    covariate1: str | int,
    covariate2: str | int | None = None,
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot every factor against one or two covariates.

    Args:
        model: The PRISMO model.
        covariate1: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
        covariate2: The first covariate to plot against. Can be an integer index or the covariate name, if the covariates are named.
            If `None`, only one covariate will be plotted.
        figsize: Figure size in inches.
    """
    return _plot_factors_covariate(model, covariate1, covariate2, gp=False, figsize=figsize)


def plot_gp_covariate(
    model: PRISMO,
    ci_opacity: float = 0.3,
    group: Literal["facet", "color"] = "facet",
    color: str = "black",
    figsize: tuple[float, float] | None = None,
) -> p9.ggplot:
    """Plot the GP posterior mean for each factor in each group at the data covariate locations.

    If the model covariates are 2D, plot the covariate on X and Y and encode the GP posterior mean with color.
    If the model covariates are 1D, plot the covariate on X and the GP posterior mean and 95% confidence interval on Y.

    Args:
        model: The PRISMO model.
        ci_opacity: Opacity of the 95% CI band. Only relevant for 1D covariates.
        group: Whether to encode the sample groups by color or by faceting. Only relevant for 1D covariates.
        color: Color of the line and CI and. Only relevant for 1D covariates and `group="facet"`.
        figsize: Figure size in inches.
    """
    covariates = model.covariates
    covdim = np.unique(tuple(cov.shape[1] for cov in covariates.values()))
    if covdim.size > 1:
        raise NotImplementedError("Different groups have different covariate dimensions.")
    elif covdim[0] > 2:
        raise NotImplementedError("Cannot plot covariates with >2 dimensions.")

    covnames = [np.asarray(tuple(n[i] for n in model.covariates_names.values())) for i in range(covdim[0])]
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
        return _plot_factors_covariate(model, 0, 1, gp=True, figsize=figsize) + p9.labs(x=covnames[0], y=covnames[1])

    gp_means = model.get_gps()
    gp_stds = model.get_gps(moment="std")

    if figsize is None:
        figsize = (2 * model.n_factors, 2 * len(gp_means))

    df = []

    for group_name, covars in covariates.items():
        cdf_mean = (
            gp_means[group_name]
            .reset_index(names="sample")
            .assign(cov=covars[:, 0])
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


def plot_smoothness(model: PRISMO, figsize: tuple[float, float] = (3, 3)) -> p9.ggplot:
    """Plot the smoothness of the GP for each factor.

    Args:
        model: The PRISMO model.
        figsize: Figure size in inches.
    """
    scale = model.gp_scale
    if scale is None:
        raise ValueError("model does not have any groups with a GP prior.")
    df = pd.DataFrame(
        {"smoothness": scale, "factor": pd.Categorical(model.factor_names, categories=model.factor_names)}
    )
    plt = (
        p9.ggplot(df, p9.aes("factor", "smoothness"))
        + p9.geom_bar(stat="identity")
        + p9.labs(x="", y="Smoothness")
        + p9.scale_y_continuous(limits=(0, 1), expand=(0, 0))
        + p9.theme(figure_size=figsize, axis_text_x=p9.element_text(rotation=90))
    )
    return plt


def _prepare_weights_df(
    model: PRISMO,
    n_features: int = 10,
    views: str | Sequence[str] | None = None,
    factors: int | Sequence[int] | None = None,
):
    weights = model.get_weights(ordered=False)
    annotations = model.get_annotations(ordered=False)
    if views is None:
        views = model.view_names
    elif isinstance(views, str):
        views = [views]
    if factors is None:
        factors = np.arange(model.n_factors)
    else:
        if not isinstance(factors, Sequence) or isinstance(factors, str):
            factors = (factors,)
        if all(isinstance(factor, str) for factor in factors):
            factors = np.where(np.isin(model.factor_names, factors))[0] + 1
        factors = np.asarray(factors) - 1

    df = []
    have_annot = False

    for view in views:
        cdf = (
            weights[view]
            .iloc[factors, :]
            .reset_index(names="factor")
            .melt(id_vars="factor", var_name="feature", value_name="weight")
        )
        if view in annotations:
            cdf = pd.merge(
                cdf,
                annotations[view]
                .reset_index(names="factor")
                .melt(id_vars="factor", var_name="feature", value_name="annotation"),
                how="left",
                on=["factor", "feature"],
            )
            cdf["annotation"].fillna(False, inplace=True)
            have_annot = True
        else:
            cdf = cdf.assign(annotation=False)
        df.append(
            cdf.assign(
                view=view,
                weightabs=lambda x: x.weight.abs(),
                factor=lambda x: pd.Categorical(x.factor, categories=model.factor_names),
                inferred=lambda x: ~x.annotation & x.weightabs > 0,
            )
        )

    df = pd.concat(df, axis=0, ignore_index=False)
    return views, factors, df, have_annot


_weights_inferred_color_scale = p9.scale_color_manual(
    values=("red", "black"), breaks=(True, False), labels=("Inferred", "Annotated")
)


def plot_top_weights(
    model: PRISMO,
    n_features: int = 10,
    views: str | Sequence[str] | None = None,
    factors: int | Sequence[int] | None = None,
    figsize: tuple[int, int] = (5, 5),
    nrows: int | None = None,
    ncols: int | None = None,
) -> p9.ggplot:
    """Plot the top weights for a given factor and view.

    Args:
        model: The PRISMO model.
        n_features: Number of top features to plot.
        views: The views to consider in the ranking. If `None`, plot all views.
        factors: The factors to plot. If `None`, plot all factors.
        figsize: Figure size in inches.
        nrows: Number of rows in the faceted plot. If None, plotnine will determine automatically.
        ncols: Number of columns in the faceted plot. If None, plotnine will determine automatically.
    """
    views, factors, df, have_annot = _prepare_weights_df(model, n_features, views, factors)
    df = (
        df.groupby("factor")
        .apply(lambda x: x.iloc[x.weightabs.argsort()[-n_features:], :], include_groups=False)
        .reset_index(0)
        .reset_index(drop=True)
        .assign(weightsgn=lambda x: x.weight >= 0)
        .sort_values(["factor", "weightabs"], ascending=True)
    )
    if len(views) > 1 and (df.groupby("factor")["feature"].aggregate(lambda x: x.duplicated().sum()) > 0).any():
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
        + p9.theme(figure_size=figsize)
    )

    facet_kwargs = {"scales": "free"}
    if nrows is not None:
        facet_kwargs["nrow"] = nrows
    if ncols is not None:
        facet_kwargs["ncol"] = ncols

    plot += p9.facet_wrap("factor", **facet_kwargs)

    return plot


def plot_weights(
    model: PRISMO,
    n_features: int = 10,
    views: str | Sequence[str] | None = None,
    factors: int | str | Sequence[int] | Sequence[str] | None = None,
    pointsize: float = 2,
    figsize: tuple[int, int] | None = None,
    nrows: int | None = None,
    ncols: int | None = None,
) -> p9.ggplot:
    """Plot the weights for a given factor and view.

    Args:
        model: The PRISMO model.
        n_features: Number of top features to annotate.
        views: The views to consider in the ranking. If `None`, use all views.
        factors: The factors to plot. If `None`, plot all factors.
        pointsize: Point size for the annotated features. Points for unannotated features will be
            of size `0.25 * pointsize`.
        figsize: Figure size in inches.
        nrows: Number of rows in the faceted plot. If None, plotnine will determine automatically.
        ncols: Number of columns in the faceted plot. If None, plotnine will determine automatically.
    """
    views, factors, df, have_annot = _prepare_weights_df(model, n_features, views, factors)
    if figsize is None:
        figsize = (3 * len(factors), 3 * len(views))
        if p9.options.limitsize:
            figsize = (min(figsize[0], 25), min(figsize[1], 25))

    grp = df.groupby(["factor", "view"])
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

    if nrows is not None or ncols is not None:
        df["facet"] = df["view"] + " - " + df["factor"].astype(str)
        labeled_data["facet"] = labeled_data["view"] + " - " + labeled_data["factor"].astype(str)

        facet_kwargs = {"scales": "free_y"}
        if nrows is not None:
            facet_kwargs["nrow"] = nrows
        if ncols is not None:
            facet_kwargs["ncol"] = ncols

        plot += p9.facet_wrap("facet", **facet_kwargs)
    else:
        plot += p9.facet_grid("view", "factor", scales="free_y")

    return plot
