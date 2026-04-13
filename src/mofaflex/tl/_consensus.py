"""Consensus matrix factorization for MOFA-FLEX.

Port of the consensus NMF principles from :cite:p:`kotliar2019identifying`
(see the `cNMF package <https://github.com/dylkot/cNMF>`_) to the multi-view
MOFA-FLEX setting. The pipeline runs several MOFA-FLEX fits with different
random seeds, pools the resulting weight spectra across runs, filters outliers
via local density, clusters the survivors with k-means, takes a per-cluster
median consensus weight matrix per view, and refits the factor (usage)
matrices via non-negative least squares against the fixed consensus weights.
Reconstruction error and silhouette stability can be computed across a sweep
of factor counts to aid k selection.

Only fully non-negative fits are supported (both ``nonnegative_factors`` and
``nonnegative_weights`` set to ``True``).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import non_negative_factorization
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances
from tqdm.auto import tqdm

from .._core import MOFAFLEX
from .._core.utils import sample_all_data_as_one_batch

_logger = logging.getLogger(__name__)


@dataclass
class ConsensusResult:
    """Result of a consensus MOFA-FLEX run.

    Attributes:
        n_factors: Number of consensus factors (``k``).
        view_names: View names, in the order used for the stacked spectra.
        group_names: Group names, in the order used for the refit usage.
        consensus_weights: Median consensus weight matrices, keyed by view
            name. Each is a :class:`pd.DataFrame` of shape
            ``(n_features[view], n_factors)``.
        consensus_factors: Non-negative factor (usage) matrices obtained by
            refitting against the consensus weights, keyed by group name.
            Each is a :class:`pd.DataFrame` of shape
            ``(n_samples[group], n_factors)``.
        stability: Silhouette score of the KMeans clustering on the pooled,
            density-filtered, L2-normalized spectra. Higher is better; the
            cNMF convention uses values in ``[-1, 1]``.
        reconstruction_error: Squared Frobenius error between the data and
            the consensus reconstruction ``Z @ W.T`` summed across groups and
            views.
        local_density: Per-spectrum local density (mean distance to the
            ``local_neighborhood_size * n_runs`` nearest neighbors). Index is
            ``"run{i}_factor{j}"``.
        kept_mask: Boolean mask indicating which spectra survived local-
            density outlier filtering. Same index as ``local_density``.
        cluster_labels: KMeans cluster label for each surviving spectrum.
            Index is the subset of ``local_density.index`` where
            ``kept_mask`` is ``True``.
        per_run_seeds: Seeds used for each replicate fit.
        density_threshold: The density threshold used for outlier filtering.
    """

    n_factors: int
    view_names: tuple[str, ...]
    group_names: tuple[str, ...]
    consensus_weights: dict[str, pd.DataFrame]
    consensus_factors: dict[str, pd.DataFrame]
    stability: float
    reconstruction_error: float
    local_density: pd.Series
    kept_mask: pd.Series
    cluster_labels: pd.Series
    per_run_seeds: tuple[int, ...]
    density_threshold: float


@dataclass
class KSelectionResult:
    """Result of a consensus sweep across multiple factor counts.

    Attributes:
        results: Mapping from ``k`` to the corresponding
            :class:`ConsensusResult`.
        stability: Stability (silhouette score) per ``k``.
        reconstruction_error: Reconstruction error per ``k``.
    """

    results: dict[int, ConsensusResult]
    stability: pd.Series = field(init=False)
    reconstruction_error: pd.Series = field(init=False)

    def __post_init__(self):
        ks = sorted(self.results.keys())
        self.stability = pd.Series(
            [self.results[k].stability for k in ks], index=ks, name="stability"
        )
        self.reconstruction_error = pd.Series(
            [self.results[k].reconstruction_error for k in ks],
            index=ks,
            name="reconstruction_error",
        )

    def plot(self, figsize: tuple[float, float] = (8, 4)):
        """Plot stability and reconstruction error versus ``k``.

        Returns:
            A :class:`plotnine.ggplot` object with stability (blue) and
            reconstruction error (red) overlaid on a twin y axis-style
            facet.
        """
        import plotnine as p9

        df = pd.DataFrame(
            {
                "k": list(self.stability.index) * 2,
                "value": list(self.stability.values) + list(self.reconstruction_error.values),
                "metric": ["Stability"] * len(self.stability)
                + ["Reconstruction error"] * len(self.reconstruction_error),
            }
        )
        plot = (
            p9.ggplot(df, p9.aes(x="k", y="value"))
            + p9.geom_line()
            + p9.geom_point()
            + p9.facet_wrap("~metric", scales="free_y")
            + p9.labs(x="Number of factors (k)", y="")
            + p9.theme(figure_size=figsize)
        )
        return plot


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _instantiate_template(
    template: MOFAFLEX | Callable[[], MOFAFLEX],
) -> MOFAFLEX:
    """Return a fresh, untrained :class:`MOFAFLEX` instance from ``template``.

    Accepts either a callable that returns a new :class:`MOFAFLEX`, or an
    untrained :class:`MOFAFLEX` which will be shallow-copied via
    :func:`copy.deepcopy`. A callable is preferred since it avoids having to
    deepcopy prior/Pyro state.
    """
    import copy

    if callable(template):
        model = template()
        if not isinstance(model, MOFAFLEX):
            raise TypeError(
                f"template_factory must return a MOFAFLEX instance, got {type(model).__name__}."
            )
        if hasattr(model, "_model"):
            raise ValueError("template_factory returned an already-trained model.")
        return model

    if not isinstance(template, MOFAFLEX):
        raise TypeError(
            f"template must be a MOFAFLEX instance or a callable returning one, got {type(template).__name__}."
        )
    if hasattr(template, "_model"):
        raise ValueError(
            "template is an already-trained MOFAFLEX model. Pass a fresh model or a factory."
        )
    return copy.deepcopy(template)


def _validate_nonnegative(model: MOFAFLEX) -> None:
    """Raise a :class:`ValueError` if the fitted model is not fully non-negative.

    Checks that every group's factors and every view's weights are element-wise
    non-negative. This mirrors the precondition documented on
    :func:`fit_consensus` and catches misconfigured templates early.
    """
    factors = model.get_factors()
    weights = model.get_weights()

    for group_name, df in factors.items():
        if (df.values < 0).any():
            raise ValueError(
                f"fit_consensus requires fully non-negative factors, but group '{group_name}' "
                "contains negative values. Configure the template with nonnegative_factors=True."
            )
    for view_name, df in weights.items():
        if (df.values < 0).any():
            raise ValueError(
                f"fit_consensus requires fully non-negative weights, but view '{view_name}' "
                "contains negative values. Configure the template with nonnegative_weights=True."
            )


def _run_replicates(
    template: MOFAFLEX | Callable[[], MOFAFLEX],
    data,
    n_runs: int,
    seed: int | None,
    fit_kwargs: Mapping[str, Any] | None,
    show_progress: bool,
) -> tuple[
    list[dict[str, pd.DataFrame]],
    list[dict[str, pd.DataFrame]],
    tuple[str, ...],
    tuple[str, ...],
    int,
    tuple[int, ...],
    MOFAFLEX,
]:
    """Fit ``n_runs`` replicates with distinct seeds and collect their factors/weights.

    Returns (per_run_weights, per_run_factors, view_names, group_names,
    n_factors, seeds, last_model). The last fitted model is retained so the
    caller can reuse its :class:`MofaFlexDataset` for the downstream refit
    without re-preprocessing the data.
    """
    fit_kwargs = dict(fit_kwargs) if fit_kwargs is not None else {}
    # We manage seed/plot_data_overview ourselves.
    fit_kwargs.pop("seed", None)
    fit_kwargs.setdefault("plot_data_overview", False)

    rng = np.random.default_rng(seed)
    seeds = tuple(int(rng.integers(1, 2**31 - 1)) for _ in range(n_runs))

    per_run_weights: list[dict[str, pd.DataFrame]] = []
    per_run_factors: list[dict[str, pd.DataFrame]] = []
    view_names: tuple[str, ...] | None = None
    group_names: tuple[str, ...] | None = None
    n_factors: int | None = None
    last_model: MOFAFLEX | None = None

    iterator = range(n_runs)
    if show_progress:
        iterator = tqdm(iterator, desc="consensus replicates", total=n_runs)

    for i in iterator:
        model = _instantiate_template(template)
        model.fit(data, seed=seeds[i], **fit_kwargs)

        if i == 0:
            _validate_nonnegative(model)
            view_names = tuple(str(v) for v in model.view_names)
            group_names = tuple(str(g) for g in model.group_names)

        weights = model.get_weights()
        factors = model.get_factors()

        # Ensure columns are in a consistent factor order across runs (they
        # are by default because we don't pass ordered=True, but we freeze
        # the column order from the first run to be safe).
        if i == 0:
            n_factors = weights[next(iter(view_names))].shape[1]
            factor_cols = list(weights[next(iter(view_names))].columns)
        weights = {v: weights[v][factor_cols] for v in view_names}
        factors = {g: factors[g][factor_cols] for g in group_names}

        per_run_weights.append(weights)
        per_run_factors.append(factors)
        last_model = model

    return (
        per_run_weights,
        per_run_factors,
        view_names,
        group_names,
        int(n_factors),
        seeds,
        last_model,
    )


def _build_spectra_matrix(
    per_run_weights: Sequence[Mapping[str, pd.DataFrame]],
    view_names: Sequence[str],
    n_factors: int,
) -> tuple[np.ndarray, list[tuple[int, int]], list[str]]:
    """Build the pooled, per-view-normalized, L2-normalized spectra matrix.

    Each row is one (run, factor) pair. For each view we take the column of
    the run's weight matrix, L2-normalize it within that view (so views
    contribute comparable magnitudes to the distance metric even if their
    raw scales differ), then concatenate view slices into one long row
    vector. A final L2 normalization over the full concatenated vector
    mirrors cNMF's normalization of its H rows before clustering.

    Returns:
        S: Array of shape ``(n_runs * n_factors, sum_v p_v)``.
        row_keys: List of ``(run_index, factor_index)`` tuples aligned to
            rows of S.
        row_labels: Human-readable row labels (``"run{i}_factor{j}"``).
    """
    eps = np.finfo(np.float64).eps
    n_runs = len(per_run_weights)

    view_cols: dict[str, list[np.ndarray]] = {v: [] for v in view_names}
    row_keys: list[tuple[int, int]] = []
    row_labels: list[str] = []

    for run_idx, weights in enumerate(per_run_weights):
        # Per-view L2 normalize each column, then stack across views
        for factor_idx in range(n_factors):
            row_keys.append((run_idx, factor_idx))
            row_labels.append(f"run{run_idx}_factor{factor_idx}")
        for v in view_names:
            arr = np.asarray(weights[v].values, dtype=np.float64)  # (p_v, k)
            norms = np.linalg.norm(arr, axis=0, keepdims=True)  # (1, k)
            arr = arr / np.where(norms > eps, norms, 1.0)
            view_cols[v].append(arr)  # (p_v, k)

    # For each view, stack its per-run (p_v, k) blocks column-wise to get
    # (p_v, n_runs * k), then transpose to (n_runs * k, p_v).
    per_view_rows = []
    for v in view_names:
        stacked = np.concatenate(view_cols[v], axis=1)  # (p_v, n_runs * k)
        per_view_rows.append(stacked.T)  # (n_runs * k, p_v)
    S = np.concatenate(per_view_rows, axis=1)  # (n_runs * k, sum_p)

    # Global L2 normalization (cNMF-style).
    norms = np.linalg.norm(S, axis=1, keepdims=True)
    S = S / np.where(norms > eps, norms, 1.0)
    return S, row_keys, row_labels


def _local_density_filter(
    S: np.ndarray,
    n_runs: int,
    local_neighborhood_size: float,
    density_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-row local densities and an outlier kept mask.

    The local density of a row is the mean Euclidean distance to its
    ``nn`` nearest neighbors, where ``nn = max(1, int(local_neighborhood_size
    * n_runs))``. Rows whose density is strictly below ``density_threshold``
    are kept.

    Returns (local_density, kept_mask) with shapes ``(n_rows,)`` each.
    """
    D = euclidean_distances(S)
    nn = max(1, int(round(local_neighborhood_size * n_runs)))
    # For each row, take mean of the nn nearest *other* rows.
    sorted_D = np.sort(D, axis=1)
    # sorted_D[:, 0] is self (distance 0); nearest neighbors are columns 1..nn
    local_density = sorted_D[:, 1 : nn + 1].mean(axis=1)
    kept_mask = local_density < density_threshold
    return local_density, kept_mask


def _cluster_and_consensus(
    S_kept: np.ndarray,
    kept_row_keys: Sequence[tuple[int, int]],
    per_run_weights: Sequence[Mapping[str, pd.DataFrame]],
    view_names: Sequence[str],
    n_factors: int,
) -> tuple[dict[str, pd.DataFrame], np.ndarray, float]:
    """KMeans-cluster the kept spectra and build per-view median consensus weights.

    Cluster labels are assigned to contiguous factor IDs in the order the
    clusters first appear. The consensus weight for cluster ``c`` and view
    ``v`` is the element-wise median of the (un-normalized) columns of
    ``per_run_weights[run_idx][v]`` for all ``(run_idx, factor_idx)`` in
    that cluster.
    """
    km = KMeans(n_clusters=n_factors, n_init=10, random_state=1)
    raw_labels = km.fit_predict(S_kept)
    stability = float(silhouette_score(S_kept, raw_labels, metric="euclidean"))

    # Relabel clusters 0..k-1 in the order they first appear so downstream
    # factor numbering is deterministic across runs of the same data.
    remap: dict[int, int] = {}
    for lbl in raw_labels:
        if lbl not in remap:
            remap[lbl] = len(remap)
    labels = np.array([remap[l] for l in raw_labels], dtype=int)

    # Gather (run_idx, factor_idx) per cluster.
    cluster_members: dict[int, list[tuple[int, int]]] = {c: [] for c in range(n_factors)}
    for key, lbl in zip(kept_row_keys, labels, strict=True):
        cluster_members[int(lbl)].append(key)

    consensus_weights: dict[str, pd.DataFrame] = {}
    for v in view_names:
        feature_index = per_run_weights[0][v].index
        cols: list[np.ndarray] = []
        for c in range(n_factors):
            members = cluster_members[c]
            if not members:
                # Degenerate cluster: fall back to a zero vector. This
                # shouldn't happen in practice because KMeans always
                # produces exactly n_factors clusters, but guard anyway.
                cols.append(np.zeros(len(feature_index), dtype=np.float64))
                continue
            stacked = np.stack(
                [per_run_weights[r][v].values[:, j] for r, j in members], axis=1
            )  # (p_v, |members|)
            cols.append(np.median(stacked, axis=1))
        consensus_weights[v] = pd.DataFrame(
            np.stack(cols, axis=1),
            index=feature_index,
            columns=[f"Factor {c + 1}" for c in range(n_factors)],
        )
    return consensus_weights, labels, stability


def _get_data_matrices(
    model: MOFAFLEX,
    data,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """Pull the preprocessed data matrices and sample names from the fitted model's dataset.

    Reuses the model's own :class:`MofaFlexDataset` construction so the
    consensus refit sees exactly the same view/group layout, feature
    subsetting and layer selection that the replicate fits saw.

    Returns ``(X, sample_names)`` where ``X[group][view]`` is a dense numpy
    array aligned to the global sample/feature axes.
    """
    from scipy.sparse import issparse

    dataset = model._make_dataset(data)
    batch = dataset.__getitems__(sample_all_data_as_one_batch(dataset))["data"]

    X: dict[str, dict[str, np.ndarray]] = {}
    for group_name, views in batch.items():
        X[group_name] = {}
        for view_name, arr in views.items():
            if issparse(arr):
                arr = arr.toarray()
            arr = np.asarray(arr, dtype=np.float64)
            arr = np.nan_to_num(arr, nan=0.0)
            X[group_name][view_name] = arr

    sample_names = {g: np.asarray(s) for g, s in dataset.sample_names.items()}
    return X, sample_names


def _refit_usages(
    consensus_weights: Mapping[str, pd.DataFrame],
    X_by_group: Mapping[str, Mapping[str, np.ndarray]],
    sample_names: Mapping[str, np.ndarray],
    view_names: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Refit per-group non-negative factor matrices against fixed consensus weights.

    Solves, for each group ``g``:

        min_Z>=0  || [X_g^{v1} | X_g^{v2} | ...] - Z @ [W_cons[v1] ; ...].T ||_F^2

    by calling :func:`sklearn.decomposition.non_negative_factorization` with
    ``update_H=False``. This reconciles all views through a single shared
    factor matrix, mirroring cNMF's NNLS usage refit but generalized to the
    multi-view case.
    """
    k = next(iter(consensus_weights.values())).shape[1]
    factor_names = list(next(iter(consensus_weights.values())).columns)

    # H has shape (k, sum_v p_v); it's W_cons[v].T stacked along the feature axis.
    H_blocks = [consensus_weights[v].values.T for v in view_names]  # each (k, p_v)
    H = np.concatenate(H_blocks, axis=1).astype(np.float64)

    # Ensure H is strictly positive enough for sklearn's init checks; a tiny
    # floor prevents zero-column edge cases that would otherwise make the
    # corresponding Z column unidentifiable.
    eps = 1e-12
    H = np.where(H > 0, H, eps)

    consensus_factors: dict[str, pd.DataFrame] = {}
    for group_name, views in X_by_group.items():
        X_blocks = [np.asarray(views[v], dtype=np.float64) for v in view_names]
        X_g = np.concatenate(X_blocks, axis=1)
        # sklearn expects X >= 0 in frobenius mode. Clamp negatives to 0.
        X_g = np.where(X_g > 0, X_g, 0.0)

        # When update_H=False, sklearn reuses the supplied H and ignores any
        # initial W — it picks its own. init="custom" is required so that the
        # supplied H is respected.
        W, _, _ = non_negative_factorization(
            X=X_g,
            H=H,
            n_components=k,
            init="custom",
            update_H=False,
            solver="cd",
            beta_loss="frobenius",
            tol=1e-4,
            max_iter=500,
        )
        consensus_factors[group_name] = pd.DataFrame(
            W,
            index=sample_names[group_name],
            columns=factor_names,
        )

    return consensus_factors


def _reconstruction_error(
    X_by_group: Mapping[str, Mapping[str, np.ndarray]],
    consensus_weights: Mapping[str, pd.DataFrame],
    consensus_factors: Mapping[str, pd.DataFrame],
    view_names: Sequence[str],
) -> float:
    """Sum of squared Frobenius errors over all (group, view) pairs."""
    total = 0.0
    for group_name, Z_df in consensus_factors.items():
        Z = Z_df.values
        for v in view_names:
            W = consensus_weights[v].values
            X = X_by_group[group_name][v]
            X = np.where(np.isfinite(X), X, 0.0)
            X = np.where(X > 0, X, 0.0)
            recon = Z @ W.T
            diff = X - recon
            total += float(np.sum(diff * diff))
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_consensus(
    template_factory: Callable[[], MOFAFLEX] | MOFAFLEX,
    data,
    *,
    n_runs: int = 30,
    density_threshold: float = 0.5,
    local_neighborhood_size: float = 0.30,
    seed: int | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    show_progress: bool = True,
) -> ConsensusResult:
    """Fit MOFA-FLEX several times and build a consensus factorization.

    This is the multi-view analog of :cite:p:`kotliar2019identifying`
    consensus NMF. Given a factory that produces fresh, untrained, fully
    non-negative MOFA-FLEX templates, this function:

    1. Runs ``n_runs`` independent fits with distinct random seeds.
    2. Pools each run's per-view weight matrices into a single stacked
       ``(n_runs * k, sum_v n_features[v])`` spectra matrix, with per-view
       L2 normalization so that views contribute comparably to Euclidean
       distances.
    3. Filters outlier spectra whose mean distance to their
       ``local_neighborhood_size * n_runs`` nearest neighbors exceeds
       ``density_threshold``.
    4. Clusters the survivors with KMeans (``k`` clusters, ``n_init=10``,
       ``random_state=1``) and computes element-wise median consensus
       weights per cluster and view.
    5. Refits the per-group factor (usage) matrices against the fixed
       consensus weights using non-negative least squares.

    Args:
        template_factory: A callable returning a fresh, untrained
            :class:`~mofaflex.MOFAFLEX` instance, or an untrained
            :class:`~mofaflex.MOFAFLEX` instance which will be deep-copied
            for each replicate. The template must be configured for fully
            non-negative inference (``nonnegative_factors=True`` and
            ``nonnegative_weights=True``).
        data: The data to fit, in any form accepted by
            :meth:`MOFAFLEX.fit <mofaflex.MOFAFLEX.fit>`.
        n_runs: Number of independent replicate fits.
        density_threshold: Strict upper bound on per-spectrum local density
            used for outlier filtering. Smaller means stricter filtering.
            Default 0.5 matches the cNMF default.
        local_neighborhood_size: Fraction of ``n_runs`` used as the nearest-
            neighbor count for local density. Default 0.30 matches cNMF.
        seed: Seed for drawing the per-replicate seeds. If ``None``, a
            fresh :class:`numpy.random.Generator` is used.
        fit_kwargs: Extra keyword arguments forwarded to
            :meth:`MOFAFLEX.fit <mofaflex.MOFAFLEX.fit>`. ``seed`` and
            ``plot_data_overview`` are managed internally and will be
            overridden if present.
        show_progress: Whether to show a :mod:`tqdm` progress bar across
            replicates.

    Returns:
        A :class:`ConsensusResult` with consensus weights, refitted factors,
        stability, reconstruction error and the diagnostic density/cluster
        bookkeeping.

    Raises:
        ValueError: If the fitted model is not fully non-negative, if
            ``n_runs < 2``, or if density filtering removes too many
            spectra to form ``k`` clusters.
    """
    if n_runs < 2:
        raise ValueError(f"n_runs must be at least 2 to form consensus, got {n_runs}.")

    (
        per_run_weights,
        per_run_factors,
        view_names,
        group_names,
        n_factors,
        seeds,
        last_model,
    ) = _run_replicates(
        template_factory, data, n_runs, seed, fit_kwargs, show_progress
    )

    S, row_keys, row_labels = _build_spectra_matrix(
        per_run_weights, view_names, n_factors
    )

    local_density_arr, kept_mask_arr = _local_density_filter(
        S, n_runs, local_neighborhood_size, density_threshold
    )

    local_density = pd.Series(local_density_arr, index=row_labels, name="local_density")
    kept_mask = pd.Series(kept_mask_arr, index=row_labels, name="kept")

    n_kept = int(kept_mask_arr.sum())
    if n_kept < n_factors:
        raise ValueError(
            f"Density filtering left only {n_kept} spectra, fewer than n_factors={n_factors}. "
            "Try relaxing `density_threshold` or increasing `n_runs`."
        )

    kept_row_keys = [row_keys[i] for i, kept in enumerate(kept_mask_arr) if kept]
    S_kept = S[kept_mask_arr]

    consensus_weights, cluster_labels_arr, stability = _cluster_and_consensus(
        S_kept, kept_row_keys, per_run_weights, view_names, n_factors
    )
    cluster_labels = pd.Series(
        cluster_labels_arr,
        index=[row_labels[i] for i, kept in enumerate(kept_mask_arr) if kept],
        name="cluster",
    )

    X_by_group, sample_names = _get_data_matrices(last_model, data)
    consensus_factors = _refit_usages(
        consensus_weights, X_by_group, sample_names, view_names
    )
    reconstruction_error = _reconstruction_error(
        X_by_group, consensus_weights, consensus_factors, view_names
    )

    return ConsensusResult(
        n_factors=n_factors,
        view_names=view_names,
        group_names=group_names,
        consensus_weights=consensus_weights,
        consensus_factors=consensus_factors,
        stability=stability,
        reconstruction_error=reconstruction_error,
        local_density=local_density,
        kept_mask=kept_mask,
        cluster_labels=cluster_labels,
        per_run_seeds=seeds,
        density_threshold=density_threshold,
    )


def k_selection(
    template_factory: Callable[[int], MOFAFLEX],
    data,
    k_values: Sequence[int],
    *,
    n_runs: int = 20,
    density_threshold: float = 0.5,
    local_neighborhood_size: float = 0.30,
    seed: int | None = None,
    fit_kwargs: Mapping[str, Any] | None = None,
    show_progress: bool = True,
) -> KSelectionResult:
    """Sweep consensus fits across multiple factor counts to aid k selection.

    For each ``k`` in ``k_values``, calls :func:`fit_consensus` with the
    template produced by ``template_factory(k)`` and collects stability
    and reconstruction error into a :class:`KSelectionResult`.

    Args:
        template_factory: A callable ``k -> MOFAFLEX`` that returns a fresh,
            untrained template configured with that many factors and
            non-negative priors.
        data: The data to fit.
        k_values: Factor counts to evaluate.
        n_runs: Number of replicate fits per ``k``.
        density_threshold: See :func:`fit_consensus`.
        local_neighborhood_size: See :func:`fit_consensus`.
        seed: Seed for the per-replicate RNG. A distinct derived seed is
            used per ``k`` so sweeps are reproducible.
        fit_kwargs: Extra keyword arguments forwarded to
            :meth:`MOFAFLEX.fit <mofaflex.MOFAFLEX.fit>`.
        show_progress: Whether to show per-replicate progress bars.

    Returns:
        A :class:`KSelectionResult` containing the per-``k`` consensus
        results plus aggregate stability and reconstruction error series.
    """
    if len(k_values) < 1:
        raise ValueError("k_values must contain at least one value.")

    rng = np.random.default_rng(seed)
    results: dict[int, ConsensusResult] = {}
    for k in k_values:
        k_seed = int(rng.integers(1, 2**31 - 1))
        factory = (lambda k=k: template_factory(k))  # noqa: E731
        results[int(k)] = fit_consensus(
            factory,
            data,
            n_runs=n_runs,
            density_threshold=density_threshold,
            local_neighborhood_size=local_neighborhood_size,
            seed=k_seed,
            fit_kwargs=fit_kwargs,
            show_progress=show_progress,
        )

    return KSelectionResult(results=results)
