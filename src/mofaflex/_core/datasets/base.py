from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from types import FunctionType, MethodType
from typing import Any, Concatenate, Literal, TypeAlias, TypeVar, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from numpy.typing import NDArray
from scipy.sparse import sparray, spmatrix
from torch.utils.data import Dataset

from ..utils import checked_baseclass

T = TypeVar("T")
ApplyCallable: TypeAlias = Callable[Concatenate[AnnData, str, str, ...], T]
ApplyToCallable: TypeAlias = Callable[Concatenate[AnnData, str, ...], T]


class Preprocessor:
    """Base class for data preprocessors."""

    def __call__(
        self,
        arr: NDArray | sparray | spmatrix,
        nonmissing_samples: NDArray[bool] | slice,
        nonmissing_features: NDArray[bool] | slice,
        group: str,
        view: str,
    ) -> tuple[NDArray | sparray | spmatrix, NDArray[int] | slice, NDArray[int] | slice]:
        """Will be called by subclasses of MofaFlexDataset on each minibatch.

        Args:
            arr: The data for one group and view.
            nonmissing_samples: Index array mapping global samples to the current minibatch.
            nonmissing_features: Index array mapping global features to the current minibatch.
            group: The group name.
            view: The view name.

        Returns:
            An array containing preprocessed data, a 1D array or slice indicating which global samples are not missing,
            and a 1D array or slice indicating which global features are not missing. If the preprocessed data is a
            subset of the input data, the nonmissing arrays will be susetted accordingly. In this implementation,
            returns the unmodified input.
        """
        return arr, nonmissing_samples, nonmissing_features


@checked_baseclass(required_init_kwargs=("sample_names", "feature_names"), required_init_kkwargs=True, registry="set")
class MofaFlexDataset(Dataset, ABC):
    """Base class for MOFA-FLEX datasets, compatible with the PyTorch dataloader interface.

    Key concepts:
        We distinguish between global and local samples/features. Global samples are the union of samples from all
        groups and views. Local samples correspond to one view in one group. Global samples may be differently
        ordered than local samples and may contain samples not present in individual views.

    Requirements for subclasses:
        Subclasses must implement methods to align local samples to global samples and vice versa. Similarly for features.

        The constructor of subclasses must take a **kwargs argument which is ignored. This ensures that users can
        simply call `MofaFlexDataset(data, args)`, where args may be a union of arguments suitable for different
        data types, only a subset of which will be used by the concrete Dataset. Subclasses should also force all
        constructor arguments except for the first (which should be the data) to be keyword arguments. Subclass
        constructors must also take arguments `sample_names` and `feature_names`, both of which default to `None`.
        If given, they specify the global sample and feature names, respectively, to align the data to.

    Preprocessor interface:
        The preprocessor must be able to process an entire minibatch. If it is a function, it will have four functions
        injected into its global namespace: `align_global_array_to_local`, `align_local_array_to_global`, `map_global_indices_to_local`,
        and `map_local_indices_to_global`. These are methods of the given MofaFlexDataset instance, see their documentation
        for how to use them. If the preprocessor is an instance of a class, these four functions will be added to its
        instance attributes. The preprocessor must accept five arguments: A (possibly sparse) array with data, a 1D
        index array indicating which global samples correspond to which samples in the current minibatch, a 1D index
        array indicating which global features correspond to features in the current minibatch, the group name, and
        the view name. If the preprocessor subsets samples or features, it must correspondingly subset the index arrays.
        Instead of index arrays, `slice(None)` may be passed. The preprocessor must return a 3-tuple containing the
        preprocessed data and the index arrays/slices.

    Args:
        data: The data. This will be stored as `self._data` and can be accessed and manipulaed by subclasses.
        preprocessor: A preprocessor. If None, will use the default preprocessor that does not apply any preprocessing.
        cast_to: Data type to cast the data to. If `None`, no casting shall be performed.
    """

    def __init__(
        self,
        data,
        *,
        preprocessor: Preprocessor | None = None,
        cast_to: Union[np.ScalarType] | None = np.float32,  # noqa UP007
    ):
        super().__init__()

        self._data = data
        self.preprocessor = preprocessor
        self._cast_to = cast_to

    def __new__(cls, data, *args, **kwargs):
        if cls != __class__:
            return super().__new__(cls)
        for subcls in __class__._registry:
            if subcls._accepts_input(data):
                return subcls.__new__(subcls, data, *args, **kwargs)
        raise NotImplementedError("Input data type not recognized.")

    @staticmethod
    @abstractmethod
    def _accepts_input(data) -> bool:
        """Determines if `data` can be handled by the given Dataset.

        Returns:
            `True` if the Dataset accepts this particular input.`False` otherwise, e.g. if the type of `data` cannot
            be processed by the Dataset.
        """
        pass

    @property
    def preprocessor(self) -> Preprocessor:
        """The preprocessor."""
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, preproc: Preprocessor):
        self._preprocessor = self._inject_alignment_functions(preproc) if preproc is not None else Preprocessor()

    @property
    def cast_to(self) -> Union[np.ScalarType] | None:  # noqa UP007
        """The data type to cast to."""
        return self._cast_to

    @property
    @abstractmethod
    def n_features(self) -> dict[str, int]:
        """Number of features in each view."""
        pass

    @property
    @abstractmethod
    def n_samples(self) -> dict[str, int]:
        """Number of samples in each group."""
        pass

    @property
    def n_samples_total(self) -> int:
        """Total number of samples."""
        return sum(self.n_samples.values())

    @property
    def n_features_total(self) -> int:
        """Total number of features."""
        return sum(self.n_features.values())

    @property
    @abstractmethod
    def view_names(self) -> NDArray[str]:
        """View names."""
        pass

    @property
    @abstractmethod
    def group_names(self) -> NDArray[str]:
        """Group names."""
        pass

    @property
    @abstractmethod
    def sample_names(self) -> dict[str, NDArray[str]]:
        """Sample names for each group."""
        pass

    @property
    @abstractmethod
    def feature_names(self) -> dict[str, NDArray[str]]:
        """Feature names for each view."""
        pass

    def get_names(self, axis: Literal[0, 1, "samples", "features"]) -> dict[str, NDArray[str]]:
        if axis in (0, "samples"):
            return self.sample_names
        else:
            return self.feature_names

    def __len__(self):
        """Length of this dataset."""
        return max(self.n_samples.values())

    def __getitem__(self, idx: dict[str, int]) -> dict[str, dict]:
        """Get one sample for each group."""
        raise NotImplementedError()

    @abstractmethod
    def __getitems__(self, idx: dict[str, list[int]]) -> dict[str, dict]:
        """Get one minibatch for each group.

        The data is returned preprocessed using the set `Preprocessor`.

        Args:
            idx: Sample indices for each group.

        Returns:
            A dict with four entries: `"data"` is a nested dict with group names keys, view names as subkeys and
            Numppy arrays of observations as values. `"sample_idx"` is the sample index (the `idx` argument
            passed through). `"nonmissing_samples"` is a nested dict with group names as keys, view names as subkeys
            and NumPy index arrays indicating which samples **in the current minibatch** are not missing as values.
            If there are no missing samples, the value may be `slice(None)`. Similarly, `"nonmissing_features"`
            indicates which features are not missing.
        """
        pass

    @abstractmethod
    def reindex_samples(self, sample_names: dict[str, NDArray[str]] | None = None):
        """Realign the samples.

        Args:
            sample_names: Global sample names for each group. If `None`, will use the natural global alignment of the data.
        """
        pass

    @abstractmethod
    def reindex_features(self, feature_names: dict[str, NDArray[str]] | None = None):
        """Realign the features.

        Args:
            feature_names: Global feature names for each view. If `None`, will use the natural global alignment of the data.
        """
        pass

    @abstractmethod
    def align_local_array_to_global(
        self,
        arr: NDArray[T],
        group_name: str,
        view_name: str,
        align_to: Literal["samples", "features"],
        axis: int = 0,
        fill_value: np.ScalarType = np.nan,
    ) -> NDArray[T]:
        """Align an array corresponding to local samples/features to global samples/features by inserting filler values for missing observations.

        Args:
            arr: The array to align.
            group_name: Group name.
            view_name: View name.
            align_to: What to align to.
            axis: The axis to align along.
            fill_value: The value to insert for missing samples.
        """
        pass

    @abstractmethod
    def align_global_array_to_local(
        self, arr: NDArray[T], group_name: str, view_name: str, align_to: Literal["samples", "features"], axis: int = 0
    ) -> NDArray[T]:
        """Align an array corresponding to global samples/features to a local samples/features by omitting observations not present in that view.

        Args:
            arr: The array to align.
            group_name: Group name.
            view_name: View name.
            align_to: What to align to.
            axis: The axis to align along.
        """
        pass

    @abstractmethod
    def map_local_indices_to_global(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal["samples, features"]
    ) -> NDArray[int]:
        """Map indices corresponding to local samples/features to the corresponding global indices.

        Args:
            idx: The indices.
            group_name: Group name.
            view_name: View name.
            align_to: What to map to.
        """
        pass

    @abstractmethod
    def map_global_indices_to_local(
        self, idx: NDArray[int], group_name: str, view_name: str, align_to: Literal["samples, features"]
    ) -> NDArray[int]:
        """Map indices corresponding to global samples/features to the corresponding local indices.

        The returned array will have values of -1 for global indices missing in the local view.

        Args:
            idx: The indices.
            group_name: Group name.
            view_name: View name.
            align_to: What to map to.
        """
        pass

    @abstractmethod
    def get_obs(self) -> dict[str, pd.DataFrame]:
        """Get observation metadata for each group."""
        pass

    @abstractmethod
    def get_missing_obs(self) -> pd.DataFrame:
        """Determine which observations are missing where.

        Returns: A dataframe with columns `view`, `group`, `obs_name`, and `missing`. `missing` is a boolean with value `True` if
            the observation `obs_name` is missing in view `view` and group `group`, and `False` otherwise.
        """
        pass

    def get_covariates(
        self,
        axis: Literal[0, 1, "samples", "features"],
        key: str | dict[str, str] | None = None,
        mkey: str | dict[str, str] | None = None,
        filter_names: str | Sequence[str] | None = None,
        fill_value: Callable[[np.dtype | pd.api.extensions.ExtensionDtype], Union[*np.ScalarType]] = lambda _: pd.NA,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Get the covariates for each group (if axis in (0, "samples")) or view (if axis in (1, "features")).

        Args:
            axis: The covariate axis
            key: Column in `.obs` or `.var` for each group/view containing the covariate.
            mkey: Key in `.obsm` or `.varm` for each group/view containing the covariates.
            filter_names: List of groups (for `axis==0`) or views (for `axis==1`) to include. If `None`, will include all groups/views.
            fill_value: Function returning the alignment fill value (see `align_local_array_to_global`) for a given array dtype.
        """
        if axis in (0, "samples"):
            names = self.group_names
            axis = 0
        else:
            names = self.view_names
            axis = 1

        if key is None:
            key = {}
        elif isinstance(key, str):
            key = dict.fromkeys(names, key)
        if mkey is None:
            mkey = {}
        elif isinstance(mkey, str):
            mkey = dict.fromkeys(names, mkey)
        if isinstance(filter_names, str):
            filter_names = (filter_names,)
        return self._get_covariates(axis, key, mkey, filter_names, fill_value)

    @abstractmethod
    def _get_covariates(
        self,
        axis: int,
        key: Mapping[str, str],
        mkey: Mapping[str, str],
        filter_names: Sequence[str] | None,
        fill_value: Callable[[np.dtype], Union[*np.ScalarType]],
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Get the covariates for each group/view.

        This method is called by `get_covariates`.
        """
        pass

    def apply(
        self,
        func: ApplyCallable[T],
        by_group: bool = True,
        by_view: bool = True,
        group_kwargs: dict[str, dict[str, Any]] | None = None,
        view_kwargs: dict[str, dict[str, Any]] | None = None,
        group_view_kwargs: dict[str, dict[str, dict[str, Any]]] | None = None,
        filter_groups: Sequence[str] | str | None = None,
        filter_views: Sequence[str] | str | None = None,
        **kwargs,
    ) -> dict[str, dict[str, T]] | dict[str, T]:
        """Apply a function to each group and/or view.

        If `func` is a function, it will have four functions injected into its global namespace: `align_global_array_to_local`,
        `align_local_array_to_global`, `map_global_indices_to_local`, and `map_local_indices_to_global`. These are methods of the
        given MofaFlexDataset instance, see their documentation for how to use them. If `func` is an instance of a class, these four
        functions will be added to its instance attributes.

        If `by_group=True` and `by_view=True`, the `AnnData` object passed to `func` will **not** have its samples and features
        aligned to the global samples/features, respectively. It is up to `func` to align when necessary using the provided functions.

        If `by_group=False`, a 1D numpy array containing the group name for each sample will be passed as second argument to `func`.
        Similarly, if `by_view=False`, a 1D numpy array containing the view name for each feature will be passed as third argument
        to `func`.

        The data contained in the passed AnnData object may be of any type that AnnData supports, not necessarily plain NumPy arrays.
        It is recommended to use the array-api-compat package to properly handle different data types.

        Args:
            func: The function to apply. The function will be passed an `AnnData` object, the group name, and the view name as the first
                three arguments.
            by_group: Whether to apply the function to each group individually or to all groups at once.
            by_view: Whether to apply the function to each view individually or to all views at once.
            group_kwargs: Additional arguments to pass to `func` for each group. The outer dict contains the argument name as key, the inner
                dict contains the value of that argument for each group. If the inner dict is missing a group, `None` will be used as the
                value of that argument for the group. Ignored if `by_group=False`.
            view_kwargs: Additional arguments to pass to `func` for each view. The outer dict contains the argument name as key, the inner
                dict contains the value of that argument for each view. If the inner dict is missing a view, `None` will be used as the
                value of that argument for the view. Ignored if `by_view=False`.
            group_view_kwargs: Additional arguments to pass to `func` for each combination of group and view. The outer dict contains the
                argument name as key, the first inner dict has groups as keys and the second inner dict has views as keys. If a group is missing
                from the outer dict or a view is missing from the inner dict, `None` will be used as the value of that argument for all views
                in the group or for the view, respectively. Ignored if `by_group=False`.
            filter_groups: List of groups to apply to. Ignored if `by_group=False` and `by_view=True`. Defaults to all groups.
            filter_views: List of views to apply to. Ignored if `by_group=True`` and `by_view=False`. Defaults to all views
            **kwargs: Additional arguments to pass to `func`.

        Returns:
            Nested dict with the return value of `func` for each group and view.
        """
        if not by_group and not by_view:
            raise NotImplementedError("At least one of `by_group` and `by_view` must be `True`.")

        if group_kwargs is None:
            group_kwargs = {}
        elif not by_group:
            raise ValueError("You cannot specify group_kwargs with `by_group=False`.")

        if view_kwargs is None:
            view_kwargs = {}
        elif not by_view:
            raise ValueError("You cannot specify view_kwargs with `by_view=False`.")

        if group_view_kwargs is None:
            group_view_kwargs = {}
        elif not by_group:
            raise ValueError("You cannot specify group_view_kwargs with `by_group=False`.")

        if isinstance(filter_groups, str):
            filter_groups = (filter_groups,)
        if isinstance(filter_views, str):
            filter_views = (filter_views,)
        func = self._inject_alignment_functions(func)
        if by_group and by_view:
            group_names = set(self.group_names) & set(filter_groups) if filter_groups is not None else self.group_names
            view_names = set(self.view_names) & set(filter_views) if filter_views is not None else self.view_names
            ckwargs = defaultdict(lambda: defaultdict(dict))

            for argname, gkwargs in group_kwargs.items():
                for group_name in group_names:
                    for view_name in view_names:
                        ckwargs[group_name][view_name][argname] = gkwargs.get(group_name, None)

            for argname, vkwargs in view_kwargs.items():
                for group_name in group_names:
                    for view_name in view_names:
                        ckwargs[group_name][view_name][argname] = vkwargs.get(view_name, None)

            for argname, gvkwargs in group_view_kwargs.items():
                for group_name in group_names:
                    gkwargs = gvkwargs.get(group_name, {})
                    for view_name in view_names:
                        ckwargs[group_name][view_name][argname] = gkwargs.get(view_name, None)

            return self._apply_by_group_view(func, group_names, view_names, ckwargs, **kwargs)
        else:
            argsdict = group_kwargs if by_group else view_kwargs
            if by_group:
                names = set(self.group_names) & set(filter_groups) if filter_groups is not None else self.group_names
            else:
                names = set(self.view_names) & set(filter_views) if filter_views is not None else self.view_names
            ckwargs = defaultdict(dict)
            for argname, vkwargs in argsdict.items():
                for name in names:
                    ckwargs[name][argname] = vkwargs.get(name, None)
            return (
                self._apply_by_group(func, names, ckwargs, **kwargs)
                if by_group
                else self._apply_by_view(func, names, ckwargs, **kwargs)
            )

    def apply_to_view(
        self, view_name: str, func: ApplyToCallable[T], group_kwargs: dict[str, dict[str, Any]] | None = None, **kwargs
    ) -> dict[str, T]:
        """Apply a function to each group for a given view.

        If `func` is a function, it will have four functions injected into its global namespace: `align_global_array_to_local`,
        `align_local_array_to_global`, `map_global_indices_to_local`, and `map_local_indices_to_global`. These are methods of the
        given MofaFlexDataset instance, see their documentation for how to use them. If `func` is an instance of a class, these four
        functions will be added to its instance attributes.

        The `AnnData` object passed to `func` will **not** have its samples and features aligned to the global samples/features,
        respectively. It is up to `func` to align when necessary using the provided functions.

        The data contained in the passed AnnData object may be of any type that AnnData supports, not necessarily plain NumPy arrays.
        It is recommended to use the array-api-compat package to properly handle different data types.

        Args:
            view_name: The name of the view to apply `func` to.
            func: The function to apply. The function will be passed an `AnnData` object and the group name as the first two arguments.
            group_kwargs: Additional arguments to pass to `func` for each group. The outer dict contains the argument name as key, the inner
                dict contains the value of that argument for each group. If the inner dict is missing a group, `None` will be used as the
                value of that argument for the group.
            **kwargs: Additional arguments to pass to `func`.

        Returns:
            dict with the return value of `func` for each group.
        """
        if group_kwargs is None:
            group_kwargs = {}

        func = self._inject_alignment_functions(func)
        ckwargs = defaultdict(lambda: defaultdict(dict))
        for argname, gkwargs in group_kwargs.items():
            for group_name in self.group_names:
                ckwargs[group_name][argname] = gkwargs.get(group_name, None)
        return self._apply_to_view(view_name, func, ckwargs, **kwargs)

    def apply_to_group(
        self, group_name: str, func: ApplyToCallable[T], view_kwargs: dict[str, dict[str, Any]] | None = None, **kwargs
    ) -> dict[str, T]:
        """Apply a function to each view for a given group.

        If `func` is a function, it will have four functions injected into its global namespace: `align_global_array_to_local`,
        `align_local_array_to_global`, `map_global_indices_to_local`, and `map_local_indices_to_global`. These are methods of the
        given MofaFlexDataset instance, see their documentation for how to use them. If `func` is an instance of a class, these four
        functions will be added to its instance attributes.

        The `AnnData` object passed to `func` will **not** have its samples and features aligned to the global samples/features,
        respectively. It is up to `func` to align when necessary using the provided functions.

        The data contained in the passed AnnData object may be of any type that AnnData supports, not necessarily plain NumPy arrays.
        It is recommended to use the array-api-compat package to properly handle different data types.

        Args:
            group_name: The name of the group to apply `func` to.
            func: The function to apply. The function will be passed an `AnnData` object and the view name as the first two arguments.
            view_kwargs: Additional arguments to pass to `func` for each view. The outer dict contains the argument name as key, the inner
                dict contains the value of that argument for each view. If the inner dict is missing a view, `None` will be used as the
                value of that argument for the view.
            **kwargs: Additional arguments to pass to `func`.

        Returns:
            dict with the return value of `func` for each view.
        """
        if view_kwargs is None:
            view_kwargs = {}

        func = self._inject_alignment_functions(func)
        ckwargs = defaultdict(lambda: defaultdict(dict))
        for argname, vkwargs in view_kwargs.items():
            for view_name in self.view_names:
                ckwargs[view_name][argname] = vkwargs.get(view_name, None)
        return self._apply_to_group(group_name, func, ckwargs, **kwargs)

    @abstractmethod
    def _apply_to_view(
        self, view_name: str, func: ApplyToCallable[T], vkwargs: dict[str, dict[str, Any]], **kwargs
    ) -> dict[str, T]:
        pass

    @abstractmethod
    def _apply_to_group(
        self, group_name: str, func: ApplyToCallable[T], gkwargs: dict[str, dict[str, Any]], **kwargs
    ) -> dict[str, T]:
        pass

    @abstractmethod
    def _apply_by_group(
        self, func: ApplyCallable[T], group_names: Sequence[str], gvkwargs: dict[str, dict[str, Any]], **kwargs
    ) -> dict[str, T]:
        pass

    @abstractmethod
    def _apply_by_view(
        self, func: ApplyCallable[T], view_names: Sequence[str], gkwargs: dict[str, dict[str, Any]], **kwargs
    ) -> dict[str, T]:
        pass

    @abstractmethod
    def _apply_by_group_view(
        self,
        func: ApplyCallable[T],
        group_names: Sequence[str],
        view_names: Sequence[str],
        vkwargs: dict[str, dict[str, dict[str, Any]]],
        **kwargs,
    ) -> dict[str, dict[str, T]]:
        pass

    def _inject_alignment_functions(self, func: Callable):
        if isinstance(func, FunctionType | MethodType):
            func.__globals__["align_global_array_to_local"] = self.align_global_array_to_local
            func.__globals__["align_local_array_to_global"] = self.align_local_array_to_global
            func.__globals__["map_global_indices_to_local"] = self.map_global_indices_to_local
            func.__globals__["map_local_indices_to_global"] = self.map_local_indices_to_global
        else:
            func.align_global_array_to_local = self.align_global_array_to_local
            func.align_local_array_to_global = self.align_local_array_to_global
            func.map_global_indices_to_local = self.map_global_indices_to_local
            func.map_local_indices_to_global = self.map_local_indices_to_global
        return func
