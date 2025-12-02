# Highly inspired by https://github.com/krassowski/gsea-api
from __future__ import annotations

import io
import logging
from collections import Counter
from collections.abc import Collection, Iterable
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class FeatureSet:
    """Class for storing a single set of features (genes).

    This class stores a single set of features (genes) and provides set operations for intersection, union, and difference.

    Attributes:
        features (frozenset): The set of features in the feature set.
        name (str): The name of the feature set.
        description (str): A description of the feature set.

    Notes:
        If the feature set is empty, a warning is raised.
        If the collection of features contains duplicates, a warning is raised.
    """

    def __init__(self, features: Collection[str], name: str, description: str = ""):
        self.name = name
        self.features = frozenset(features)
        self.description = description

        if self.empty:
            logger.warning(f"FeatureSet {name!r} is empty.")

        redundant_features = None

        if len(features) != len(self.features):
            redundant_features = {feature: count for feature, count in Counter(features).items() if count > 1}

            logger.warning(
                f"FeatureSet {name!r} received a non-unique "
                f"collection of features; redundant features: {redundant_features}"
            )

        self.redundant_features = redundant_features

    @property
    def empty(self):
        """Check if the feature set is empty."""
        return len(self) == 0

    def __len__(self):
        return len(self.features)

    def __repr__(self) -> str:
        features = ": " + ", ".join(sorted(self.features)) if len(self.features) < 5 else ""
        return f"<FeatureSet {self.name!r} with {len(self)} features{features}>"

    def __iter__(self) -> Iterable[str]:
        return iter(self.features)

    def __eq__(self, other: FeatureSet) -> bool:
        return self.features == other.features

    def __hash__(self) -> int:
        return hash(self.features)

    def __and__(self, other: FeatureSet) -> FeatureSet:
        return FeatureSet(self.features & other.features, name=f"{self.name}&{other.name}")

    def __or__(self, other: FeatureSet) -> FeatureSet:
        return FeatureSet(self.features | other.features, name=f"{self.name}|{other.name}")

    def __add__(self, other: FeatureSet) -> FeatureSet:
        return self.__or__(other)

    def subset(self, features: Iterable[str]) -> FeatureSet:
        """Subset features from a feature set.

        Args:
            features: Features to subset.
        """
        return FeatureSet(self.features & set(features), name=self.name)


class FeatureSets:
    """Class for storing a collection of feature sets (see FeatureSet).

    This class stores a collection of feature sets and provides set operations for intersection, union, and difference.

    Attributes:
        feature_sets (frozenset): The collection of feature sets.
        name (str): The name of the feature set collection.
        remove_empty (bool): Whether to remove empty feature sets.
    """

    def __init__(self, feature_sets: Collection[FeatureSet], name: str = "UNL", remove_empty: bool = True):
        self.name = name

        if remove_empty:
            feature_sets = {feature_set for feature_set in feature_sets if not feature_set.empty}

        redundant_feature_sets = None

        if len(set(feature_sets)) != len(feature_sets):
            redundant_feature_sets = {
                feature_set: count for feature_set, count in Counter(feature_sets).items() if count > 1
            }

            logger.warning(
                f"FeatureSets {name!r} received a non-unique "
                "collection of feature sets; redundant feature sets: "
                f"{redundant_feature_sets}"
            )
        self.feature_sets = frozenset(feature_sets)
        self.redundant_feature_sets = redundant_feature_sets

    @property
    def empty(self):
        """Check if the feature set collection is empty."""
        return len(self) == 0

    @property
    def median_size(self) -> int:
        """Return the median size of the feature sets."""
        return int(np.median([len(fs) for fs in self.feature_sets]))

    @property
    def features(self) -> frozenset:
        """Return the union of all features in the feature sets."""
        return frozenset().union(*(fs.features for fs in self.feature_sets))

    @property
    def feature_set_by_name(self) -> dict:
        """Return a dictionary of feature set names (key) to feature sets (value)."""
        return {feature_set.name: feature_set for feature_set in self.feature_sets}

    def __getitem__(self, name: str) -> FeatureSet:
        return self.feature_set_by_name[name]

    def __len__(self):
        return len(self.feature_sets)

    def __iter__(self) -> Iterable[FeatureSet]:
        return iter(self.feature_sets)

    def __repr__(self) -> str:
        feature_sets = (
            ": " + ", ".join(sorted({fs.name for fs in self.feature_sets})) if len(self.feature_sets) < 5 else ""
        )
        return f"<FeatureSets {self.name!r} with {len(self)} " + f"feature sets{feature_sets}>"

    def __eq__(self, other: FeatureSets) -> bool:
        return self.feature_sets == other.feature_sets

    def __hash__(self) -> int:
        return hash(self.feature_sets)

    def __and__(self, other: FeatureSets) -> FeatureSets:
        return FeatureSets(name=f"{self.name}&{other.name}", feature_sets=self.feature_sets & other.feature_sets)

    def __or__(self, other: FeatureSets) -> FeatureSets:
        return FeatureSets(name=f"{self.name}|{other.name}", feature_sets=self.feature_sets | other.feature_sets)

    def __add__(self, other: FeatureSets) -> FeatureSets:
        return self.__or__(other)

    def find(self, partial_name: str) -> FeatureSets:
        """Perform a simple search given a (partial) feature set name.

        Args:
            partial_name: Feature set (partial) name to search for.
        """
        return FeatureSets(
            {feature_set for feature_set in self.feature_sets if partial_name in feature_set.name},
            name=f"{self.name}:{partial_name}",
        )

    def remove(self, names: Iterable[str]):
        """Remove feature sets by name.

        Args:
            names: Collection of feature set names.
        """
        return FeatureSets(
            {feature_set for feature_set in self.feature_sets if feature_set.name not in names}, name=self.name
        )

    def keep(self, names: Iterable[str]):
        """Keep feature sets by name.

        Args:
            names: Collection of feature set names.
        """
        return FeatureSets(
            {feature_set for feature_set in self.feature_sets if feature_set.name in names}, name=self.name
        )

    def trim(self, min_count: int = 1, max_count: int | None = None):
        """Trim feature sets by min/max size.

        Args:
            min_count: Minimum number of features, by default 1.
            max_count: Maximum number of features, by default None.
        """
        return FeatureSets(
            {
                feature_set
                for feature_set in self.feature_sets
                if min_count <= len(feature_set) <= (max_count or len(feature_set))
            },
            name=self.name,
        )

    def subset(self, features: Iterable[str]):
        """Subset feature sets by features.

        Args:
            features: Collection of features.
        """
        return FeatureSets({feature_set.subset(set(features)) for feature_set in self.feature_sets}, name=self.name)

    def filter(
        self,
        features: Iterable[str],
        min_fraction: float = 0.5,
        min_count: int = 5,
        max_count: int | None = None,
        keep: Iterable[str] | None = None,
        subset: bool = True,
    ) -> FeatureSets:
        """Filter feature sets.

        Args:
            features: Features to filter.
            min_fraction: Mininimum portion of the feature set to be present in `features`.
            min_count: Minimum size of the intersection set between a feature set and the set of `features`.
            max_count: Maximum size of the intersection set between a feature set and the set of `features`.
            keep: Feature sets to keep regardless of the filter conditions.
            subset: Whether to subset the resulting feature sets based on `features`.

        Returns:
            Filtered feature sets.
        """
        features = set(features)

        if keep is None:
            keep = set()

        feature_set_subset = set()

        for feature_set in self.feature_sets:
            if feature_set.name in keep:
                feature_set_subset.add(feature_set)
                continue
            intersection = features & feature_set.features
            count = len(intersection)
            fraction = count / len(feature_set)
            if count >= min_count and fraction >= min_fraction and (max_count is None or count <= max_count):
                if subset:
                    feature_set = feature_set.subset(features)
                feature_set_subset.add(feature_set)

        filtered_feature_sets = FeatureSets(feature_set_subset, name=self.name)
        return filtered_feature_sets

    def to_mask(self, features: Iterable[str] | None = None, sort: bool = True) -> pd.DataFrame:
        """Convert feature sets to a mask.

        Args:
            features: Collection of features.
            sort: Sort feature sets alphabetically.

        Returns:
            Binary mask of features.
        """
        features_list = sorted(self.features) if features is None else list(features)
        feature_sets_list = sorted(self.feature_sets, key=lambda x: x.name)
        if sort:
            feature_sets_list = sorted(feature_sets_list, key=lambda fs: fs.name)
        return pd.DataFrame(
            [[feature in feature_set.features for feature in features_list] for feature_set in feature_sets_list],
            index=[feature_set.name for feature_set in feature_sets_list],
            columns=features_list,
        )

    def similarity_to_feature_sets(
        self, other: FeatureSets = None, metric: str = "jaccard", metric_kwargs: dict | None = None
    ) -> pd.DataFrame:
        """Compute similarity matrix between feature sets.

        Args:
            other: Other feature set collection, by default None.
            metric: Similarity metric, by default "jaccard".
            metric_kwargs: further arguments to `scipy.spatial.distance.cdist`

        Returns:
            Similarity matrix as 1 minus distance matrix, may lead to negative values for some distance metrics.
        """
        if metric not in ["jaccard", "cosine"]:
            logger.warning(
                f"Similarity matrix for `{metric}` might be negative. Recommended metrics are `jaccard` or `cosine`."
            )

        self_mask = self.to_mask()
        other_mask = other.to_mask() if other else self_mask

        if metric_kwargs is None:
            metric_kwargs = {}
        return 1 - pd.DataFrame(
            cdist(self_mask.to_numpy(), other_mask.to_numpy(), metric=metric, **metric_kwargs),
            index=self_mask.index,
            columns=other_mask.index,
        )

    def similarity_to_observations(self, observations: pd.DataFrame) -> pd.DataFrame:
        """Compute similarity matrix between feature sets using observations as a reference.

        Args:
            observations: Dataframe of observations.

        Returns:
            Similarity matrix as correlation matrix.
        """
        obs_mean = observations.mean(axis=1)

        mean_dists = pd.DataFrame(index=observations.index)
        for feature_set in self.feature_sets:
            col_subset = observations.columns[observations.columns.isin(feature_set.features)]
            mean_dists[feature_set.name] = (
                np.nan if len(col_subset) == 0 else observations.loc[:, col_subset].mean(axis=1) - obs_mean
            )

        return mean_dists.corr()

    def _find_similar_pairs(self, sim_matrix: pd.DataFrame, similarity_threshold: float) -> set[tuple[str, str, float]]:
        """Find similar pairs of feature sets.

        Args:
            sim_matrix: Similarity matrix.
            similarity_threshold: Similarity threshold to consider similar pairs.

        Returns:
            Similar pairs of feature sets.
        """
        pairs = set()
        visited = set()

        row_offset = 0
        for current_fs, row in sim_matrix.iterrows():
            row_offset += 1
            if row_offset >= len(row):
                break
            if current_fs in visited:
                continue
            visited.add(current_fs)
            closest_fs = row.iloc[row_offset:].idxmax()
            similarity = row[closest_fs]
            if similarity >= similarity_threshold and closest_fs not in visited:
                pairs.add((current_fs, closest_fs, similarity))
                visited.add(closest_fs)

        return pairs

    def find_similar_pairs(
        self, observations: pd.DataFrame = None, metric: str | None = None, similarity_threshold: float = 0.8
    ) -> set[tuple[str, str, float]]:
        """Find similar pairs of feature sets.

        Args:
            observations: Dataframe of observations, if provided, the similarity between feature sets is computed
                based on the correlation of the similarity from the mean of the observations in the feature set.
            metric: Similarity metric, by default "jaccard" if observations not provided.
            similarity_threshold: Similarity threshold to consider similar pairs.

        Returns:
            Similar pairs of feature sets.
        """
        if observations is None and metric is None:
            logger.warning("Neither observations nor metric is provided, using `metric=jaccard` as default.")
            metric = "jaccard"

        sim_matrix = []
        if observations is not None:
            sim_matrix.append(self.similarity_to_observations(observations))
        if metric is not None:
            sim_matrix.append(self.similarity_to_feature_sets(metric=metric))

        if observations is not None and metric is not None:
            sim_matrix[0][sim_matrix[0] < 0] = 0.0
            sim_matrix[1][sim_matrix[1] < 0] = 0.0
            sim_matrix = (2 * sim_matrix[0] * sim_matrix[1]) / (sim_matrix[0] + sim_matrix[1])
        else:
            sim_matrix = sim_matrix[0]
        return self._find_similar_pairs(sim_matrix.fillna(0.0), similarity_threshold)

    def merge_pairs(self, pairs: Iterable[tuple[str, str]]) -> FeatureSets:
        """Merge pairs of feature sets.

        Args:
            pairs: Pairs of feature sets.

        Returns:
            Merged feature sets.
        """
        names_to_remove = set()
        merged_feature_sets = set()
        for pair in pairs:
            merged_feature_sets.add(self[pair[0]] | self[pair[1]])
            names_to_remove |= {pair[0], pair[1]}

        # remove merged feature sets
        feature_sets = self.remove(names_to_remove)
        # then add merged feature sets
        feature_sets |= FeatureSets(merged_feature_sets)
        feature_sets.name = self.name
        return feature_sets

    def merge_similar(
        self,
        observations: pd.DataFrame = None,
        metric: str | None = None,
        similarity_threshold: float = 0.8,
        iteratively: bool = True,
    ) -> FeatureSets:
        """Merge similar feature sets.

        Args:
            observations: Dataframe of observations, if provided, the similarity between feature sets
                is computed based on the correlation of the similarity from the mean of the observations
                in the feature set.
            metric: Similarity metric, by default "jaccard" if observations not provided.
            similarity_threshold: Similarity threshold to consider similar pairs.
            iteratively: Whether to merge iteratively.

        Returns:
            Merged feature sets.
        """
        feature_sets = self
        while True:
            pairs = {
                (name1, name2)
                for name1, name2, _ in feature_sets.find_similar_pairs(
                    observations=observations, metric=metric, similarity_threshold=similarity_threshold
                )
            }
            stopping = ""
            if len(pairs) == 0 and iteratively:
                stopping = " Stopping..."
            logger.info(f"Found {len(pairs)} pairs to merge.{stopping}")
            feature_sets = feature_sets.merge_pairs(pairs)
            if len(pairs) == 0 or not iteratively:
                break
        return feature_sets

    def to_gmt(self, path: str | Path | io.TextIOBase):
        """Write this feature set collection to a GMT file.

        Args:
            path: Path to the output file.
        """
        if isinstance(path, io.TextIOBase):
            ctx = nullcontext(path)
        else:
            ctx = open(path, "w")
        with ctx as f:
            for feature_set in self.feature_sets:
                f.write(
                    feature_set.name + "\t" + feature_set.description + "\t" + "\t".join(feature_set.features) + "\n"
                )

    def to_dict(self) -> dict[str, Iterable[str]]:
        """Convert this feature set collection to a dictionary.

        Returns:
            Dictionary of feature sets.
        """
        return {fs.name: fs.features for fs in self.feature_sets}

    @classmethod
    def from_gmt(
        cls, path: str | Path | io.TextIOBase, name: str | None = None, remove_empty: bool = True
    ) -> FeatureSets:
        """Create a FeatureSets object from a GMT file.

        Args:
            path: Path to the GMT file.
            name: Name of the collection. Defaults to the file name.
            remove_empty: Whether to remove empty feature sets.
        """
        feature_sets = set()
        if isinstance(path, io.TextIOBase):
            ctx = nullcontext(path)
        else:
            ctx = open(path)
        with ctx as f:
            for line in f:
                fs_name, description, *features = line.strip().split("\t")
                feature_sets.add(FeatureSet(features, name=fs_name, description=description))
        return cls(feature_sets, name=name or Path(path).name, remove_empty=remove_empty)

    @classmethod
    def from_dict(cls, d: dict[str, Iterable[str]], name: str | None = None, remove_empty: bool = True) -> FeatureSets:
        """Create a FeatureSets object from a dictionary.

        Args:
            d: Dictionary of feature sets.
            name: Name of the collection.
            remove_empty: Whether to remove empty feature sets.
        """
        feature_sets = set()
        for fs_name, features in d.items():
            feature_sets.add(FeatureSet(features, name=fs_name))
        return cls(feature_sets, name=name, remove_empty=remove_empty)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        name: str | None = None,
        name_col: str = "name",
        features_col: str = "features",
        desc_col: str | None = None,
        remove_empty: bool = True,
    ) -> FeatureSets:
        """Create a FeatureSets object from a DataFrame.

        Args:
            df: DataFrame of feature sets.
            name: Name of the collection.
            name_col: Name of the column containing feature set names.
            features_col: Name of the column containing feature set features.
            desc_col: Name of the column containing feature set descriptions.
            remove_empty: Whether to remove empty feature sets.
        """
        feature_sets = set()
        for _, row in df.iterrows():
            feature_sets.add(
                FeatureSet(
                    row[features_col], name=row[name_col], description=row[desc_col] if desc_col is not None else ""
                )
            )
        return cls(feature_sets, name=name, remove_empty=remove_empty)
