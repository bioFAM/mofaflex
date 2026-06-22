import inspect
import logging
import operator
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import suppress
from functools import reduce, update_wrapper
from itertools import chain
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from anndata import AnnData
from array_api_compat import array_namespace
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroModuleList, PyroParam, pyro_method
from scipy import stats
from scipy.sparse import issparse
from sklearn.decomposition import NMF, PCA

from ..api import priors as apipriors
from ..api.utils import APIType
from ..datasets import CovariatesDataset, MofaFlexDataset, StackDataset, df_to_array, merge_covariates
from ..likelihoods.pyro import Likelihood
from ..priors import FactorPriorType, Prior, PriorDynamicAPI, WeightPriorType
from ..utils import MeanStd, PyroModuleDict, PyroParameterDict, change_pyro_plate_dim, default_torch_device
from .base import Term

_logger = logging.getLogger(__name__)


class _PriorApiProperty(NamedTuple):
    obj: Prior
    attr: str


class MofaFlex(Term):
    """A MOFA-like term representing the product of two low-rank matrices.

    The factor matrix has dimensions `n_samples` x `n_total_factors`, the weight matrix has dimensions
    `n_total_factors` x `n_features`. See :ref:`the in-depth model description <modeldescription>` for details.

    Args:
        n_factors: Number of latent factors.
        factor_prior: Factor priors for each group (if dict) or for all groups (otherwise). The dictionary keys
            may be either strings, representing individual groups, or tuples of strings, in which case the corresponding
            value will be used for all groups named in the tuple.
        weight_prior: Weight priors for each group (if dict) or for all groups (otherwise). The dictionary keys
            may be either strings, representing individual views, or tuples of strings, in which case the corresponding
            value will be used for all views named in the tuple.
        nonnegative_factors: Non-negativity constraints for factors for each group (if dict) or for all groups (if bool).
        nonnegative_weights: Non-negativity constraints for weights for each view (if dict) or for all views (if bool).
        guiding_vars_obs_keys: Keys of .obs attribute of each :class:`AnnData<anndata.AnnData>` object that contains guiding variable values.
        guiding_vars_likelihoods: Likelihood for each guiding variable (if dict) or for all guiding variables (if str).
        guiding_vars_scales: Scale for the likelihood of each guiding variable, to put more or less emphasis on them during training.
        init_factors: Initialization method for factors.
        init_scale: Initialization scale of Normal distribution for factors.
    """

    _state_attrs = (
        "_n_factors",
        "_nonnegative_factors",
        "_nonnegative_weights",
        "_guiding_vars_obs_keys",
        "_guiding_vars_likelihoods",
        "_guiding_vars_scales",
        "_guiding_vars_names",
        "_init_factors",
        "_init_scale",
        "_factor_names",
        "_factor_order",
        "_factors",
        "_weights",
    )

    def __init__(
        self,
        n_factors: int,
        factor_prior: Mapping[str | Sequence[str], FactorPriorType | apipriors.Prior]
        | FactorPriorType
        | apipriors.Prior = "Normal",
        weight_prior: Mapping[str | Sequence[str], WeightPriorType | apipriors.Prior]
        | WeightPriorType
        | apipriors.Prior = "Normal",
        nonnegative_factors: Mapping[str, bool] | bool = False,
        nonnegative_weights: Mapping[str, bool] | bool = False,
        guiding_vars_obs_keys: str | Sequence[str] | Mapping[str, Mapping[str, str]] | None = None,
        guiding_vars_likelihoods: Mapping[str, str] | Literal["Normal", "Categorical", "Bernoulli"] | None = "Normal",
        guiding_vars_scales: Mapping[str, float] | float = 1.0,
        init_factors: float | Literal["random", "orthogonal", "pca"] = "random",
        init_scale: float = 0.1,
    ):
        super().__init__()
        self._n_factors = n_factors
        self._factor_priors = factor_prior
        self._weight_priors = weight_prior
        self._nonnegative_factors = nonnegative_factors
        self._nonnegative_weights = nonnegative_weights
        self._guiding_vars_obs_keys = guiding_vars_obs_keys
        self._guiding_vars_likelihoods = guiding_vars_likelihoods
        self._guiding_vars_scales = guiding_vars_scales

        self._init_factors = init_factors
        self._init_loc = 0.0
        self._init_scale = init_scale

        self._factor_names = [f"Factor {k + 1}" for k in range(n_factors)]
        self._factor_order = np.arange(len(self._factor_names))

        self._prior_api_properties: dict[str, _PriorApiProperty] = {}

    def _results_to_df(
        self,
        results: Mapping[str, np.ndarray],
        axis: Literal[0, 1],
        ordered: bool = False,
        factors_subset: slice = slice(None),
    ):
        factor_names = self.factor_names[factors_subset]
        ret = {}
        for name, res in results.items():
            fnames = factor_names
            if ordered:
                factor_order = self.factor_order[factors_subset].copy()
                factor_order[np.argsort(factor_order)] = np.arange(len(factor_order))
                res = res[:, factor_order]
                fnames = fnames[factor_order]
            ret[name] = pd.DataFrame(
                res, index=self._sample_names[name] if axis == 0 else self._feature_names[name], columns=fnames
            )
        return ret

    def _wrap_api_method(self, axis: Literal[0, 1], prior: Prior, api: PriorDynamicAPI):
        def wrapper_func(self, *args, **kwargs):
            with torch.device(self._device):
                ret = getattr(prior, api.name)
                if api.type == APIType.method:
                    ret = ret(*args, **kwargs)
            return ret

        if not api.has_factors:
            wrapped = wrapper_func
        else:

            def wrapper_func_order(self, *args, ordered: bool = False, **kwargs):
                ret = wrapper_func(self, *args, **kwargs)
                factors_subset = getattr(prior, api.factors_subset) if api.factors_subset is not None else slice(None)
                return self._results_to_df(ret, axis, ordered, factors_subset)

            wrapped = wrapper_func_order

        return wrapped

    @staticmethod
    def _merge_api_results(results: Iterable):
        results = tuple(results)
        if (
            all(isinstance(result, Mapping) for result in results)
            and len(reduce(operator.and_, (result.keys() for result in results))) == 0
        ):
            ret = {}
            for result in results:
                ret.update(result)
            return ret
        else:
            return results

    @staticmethod
    def _wrap_list_of_wrapped_methods(methods: Iterable[Callable]):
        def wrapper_func(self, *args, **kwargs):
            return __class__._merge_api_results(method(*args, **kwargs) for method in methods)

        return wrapper_func

    def _init_api(self):
        for axis, priors in ((0, self._factor_priors), (1, self._weight_priors)):
            grouped_priors = defaultdict(list)
            for prior in priors:
                grouped_priors[prior.__class__.__name__].append(prior)
            for gpriors in grouped_priors.values():
                apis = gpriors[0].api()
                for api in apis:
                    name = _apinames[(axis, gpriors[0].__class__.__name__, api.name)]
                    self._api(name)
                    if api.type == APIType.property and not api.has_factors:
                        self._prior_api_properties[name] = _PriorApiProperty(gpriors, api.name)
                        continue
                    if len(gpriors) > 1:
                        wrapped = self._wrap_list_of_wrapped_methods(
                            self._wrap_api_method(axis, prior, api) for prior in gpriors
                        )
                    else:
                        wrapped = self._wrap_api_method(axis, gpriors[0], api)
                    dummy = getattr(self.__class__, name)
                    update_wrapper(wrapped, dummy)
                    setattr(self, name, wrapped.__get__(self))

    def __getattribute__(self, name):
        try:
            prop = super().__getattribute__("_prior_api_properties")[name]
            if len(prop.obj) > 1:
                return self._merge_api_results(getattr(obj, prop.attr) for obj in prop.obj)
            else:
                return getattr(prop.obj[0], prop.attr)
        except (KeyError, AttributeError):
            return super().__getattribute__(name)

    @Term._api
    @property
    def n_guided_factors(self) -> int:
        """Number of guided factors."""
        return len(self._guiding_vars_names)

    @property
    def _guiding_vars_factors(self) -> range:
        return range(self.n_total_factors - self.n_guided_factors, self.n_total_factors)

    @Term._api
    @property
    def n_factors(self) -> int:
        """Number of unguided factors."""
        return self._n_factors

    @Term._api
    @property
    def n_total_factors(self) -> int:
        """Total number of factors."""
        return len(self._factor_names)

    @Term._api
    @property
    def factor_names(self) -> NDArray[str | np.str_]:
        """Factor names."""
        return self._factor_names

    @property
    def component_order(self) -> NDArray[int]:
        return self._factor_order

    @component_order.setter
    def component_order(self, order: NDArray[int]):
        order = order.squeeze()
        if order.ndim != 1:
            raise ValueError(f"`order` must be 1-dimensional, got {order.ndim}-dimensional array.")
        if order.size != self.n_total_factors:
            raise ValueError(f"Wrong size of `order` argument. Need {self.n_total_factors}, got {order.size}.")
        if order.min() != 0 or order.max() != self.n_total_factors - 1 or np.unique(order).size != order.size:
            raise ValueError(f"The ordering must contain all integers in [0, {self.n_factors}).")
        self._factor_order = order

    @Term._api
    @property
    def factor_order(self) -> NDArray[int]:
        """Ordering of factors by explained variance (highest to lowest)."""
        return self.component_order

    @factor_order.setter
    def factor_order(self, order: NDArray[int]):
        self.component_order = order

    def _init(self, data: MofaFlexDataset):
        self._sample_names = data.sample_names
        self._feature_names = data.feature_names

        self._pos_transform = torch.nn.ReLU()
        for priorattr, names in zip(
            ("_factor_priors", "_weight_priors"), (data.group_names, data.view_names), strict=True
        ):
            priors = getattr(self, priorattr)
            if isinstance(priors, str):
                priorlist = [Prior(priors, names)]
            elif isinstance(priors, apipriors.Prior):
                priorlist = [priors(names)]
            else:
                priorlist = []
                for pnames, prior in priors.items():
                    if isinstance(prior, str):
                        prior = Prior(prior, pnames)
                    else:
                        prior = prior(pnames)
                    priorlist.append(prior)
            setattr(self, priorattr, PyroModuleList(priorlist))
        for prior in self._factor_priors:
            if not prior.factors_allowed():
                raise ValueError(f"The prior {prior.__class__.__name__} cannot be used for factors.")
        for prior in self._weight_priors:
            if not prior.weights_allowed():
                raise ValueError(f"The prior {prior.__class__.__name__} cannot be used for weights.")

        if isinstance(self._nonnegative_factors, bool):
            self._nonnegative_factors = dict.fromkeys(data.group_names, self._nonnegative_factors)

        if isinstance(self._nonnegative_weights, bool):
            self._nonnegative_weights = dict.fromkeys(data.view_names, self._nonnegative_weights)

        # guiding variables
        if self._guiding_vars_obs_keys is not None:
            if isinstance(self._guiding_vars_obs_keys, str):
                self._guiding_vars_obs_keys = [self._guiding_vars_obs_keys]
            if isinstance(self._guiding_vars_obs_keys, Sequence):
                self._guiding_vars_obs_keys = {
                    obs_key: dict.fromkeys(data.group_names, obs_key) for obs_key in self._guiding_vars_obs_keys
                }
            self._guiding_vars_names = list(self._guiding_vars_obs_keys.keys())
            self._factor_names = self._factor_names + self._guiding_vars_names

            if isinstance(self._guiding_vars_likelihoods, str):
                self._guiding_vars_likelihoods = dict.fromkeys(self._guiding_vars_names, self._guiding_vars_likelihoods)
        else:
            self._guiding_vars_names = []

    def _init_guiding_vars(self, data: MofaFlexDataset):
        if not isinstance(self._guiding_vars_scales, dict):
            self._guiding_vars_scales = dict.fromkeys(self._guiding_vars_names, self._guiding_vars_scales)

        total_n_features = 0.1 * data.n_features_total
        self._guiding_vars_scales = {
            name: scale * total_n_features for name, scale in self._guiding_vars_scales.items()
        }

        self._pyro_guiding_vars_likelihoods = PyroModuleDict(
            {
                guiding_var_name: Likelihood(
                    self._guiding_vars_likelihoods[guiding_var_name],
                    view_name=guiding_var_name,
                    sample_dim=self._sample_plate_dim,
                    feature_dim=self._feature_plate_dim,
                    nsamples=data.n_samples,
                    nfeatures=1,
                )
                for guiding_var_name in self._guiding_vars_names
            }
        )

        self._guiding_locs = PyroParameterDict()
        self._guiding_scales = PyroParameterDict()

        self._guiding_vars_weights_dims = {}
        for guiding_var_name in self._guiding_vars_names:
            self._guiding_vars_weights_dims[guiding_var_name] = weights_dim = max(
                self._guiding_vars_n_categories[guiding_var_name], 1
            )
            self._guiding_locs[guiding_var_name] = PyroParam(
                torch.full([weights_dim, 2], self._init_loc), constraint=constraints.real
            )
            self._guiding_scales[guiding_var_name] = PyroParam(
                torch.full([weights_dim, 2], self._init_scale), constraint=constraints.softplus_positive
            )

    def get_datasets(
        self, data: MofaFlexDataset, sample_plate_dim: int, feature_plate_dim: int
    ) -> dict[str, CovariatesDataset]:
        self._sample_plate_dim = sample_plate_dim
        self._feature_plate_dim = feature_plate_dim
        self._init(data)

        ret = defaultdict(dict)
        factor_names = set(self._factor_names)
        for axis, priors in ((0, self._factor_priors), (1, self._weight_priors)):
            for prior in priors:
                if len(new_factors := prior.extend_factors(data, axis, len(self._factor_names))) > 0:
                    prefix = "_".join(prior.names) + "_"
                    new_factors = [prefix + fname if fname in factor_names else fname for fname in new_factors]
                    self._factor_names += new_factors
                    factor_names |= set(new_factors)
        self._make_factor_names_unique()

        for prior in self._factor_priors:
            if priordsets := prior.get_datasets(data, 0, self.n_total_factors, data.n_samples):
                for dsetname, dset in priordsets.items():
                    ret[dsetname].update(dset)  # handle multiple priors of the same class with different settings
        for dsetname, dset in ret.items():
            ret[dsetname] = CovariatesDataset(dset)

        self._weight_dsets = defaultdict(dict)
        for prior in self._weight_priors:
            if priordsets := prior.get_datasets(data, 1, self.n_total_factors, data.n_features):
                for dsetname, dset in priordsets.items():
                    wdsets = self._weight_dsets[dsetname]
                    for view_name, view_dset in dset.items():
                        wdsets[view_name] = df_to_array(view_dset) if isinstance(view_dset, pd.DataFrame) else view_dset

        if self.n_guided_factors > 0:
            guiding_vars = {
                guiding_var_name: merge_covariates(data.get_covariates(0, key=obs_key))
                for guiding_var_name, obs_key in self._guiding_vars_obs_keys.items()
            }
            ret["guiding_vars"] = StackDataset(**{name: CovariatesDataset(dset) for name, dset in guiding_vars.items()})

            self._guiding_vars_n_categories = {}
            for guiding_var_name, guiding_var_likelihood in self._guiding_vars_likelihoods.items():
                if guiding_var_likelihood == "Categorical":
                    guiding_vars_categories = set()
                    # find number of unique categories across groups
                    for group_name in data.group_names:
                        guiding_vars_categories.update(guiding_vars[guiding_var_name][group_name].iloc[:, 0].to_list())
                    self._guiding_vars_n_categories[guiding_var_name] = len(guiding_vars_categories)

                else:
                    # if not categorical, set to default
                    self._guiding_vars_n_categories[guiding_var_name] = 0
        return ret

    @staticmethod
    def _init_factor_group(adata, group_name, view_name, impute_missings, initializer):
        arr = adata.X
        if issparse(arr):
            havenan = np.isnan(arr.data).any()
        else:
            xp = array_namespace(arr)
            havenan = xp.isnan(arr).any()
        if havenan:
            if impute_missings:
                from sklearn.impute import SimpleImputer

                imp = SimpleImputer(missing_values=np.nan, strategy="mean")
                arr = imp.fit_transform(arr)
            else:
                raise ValueError("Data has missing values. Please impute missings or set 'impute_missings=True'.")
        return initializer.fit_transform(arr)

    def _initialize_factors(self, data, impute_missings=True):
        init_tensor = defaultdict(dict)
        _logger.info(f"Initializing factors using '{self._init_factors}' method...")

        if not isinstance(self._init_factors, str):
            for group_name, n in data.n_samples.items():
                init_tensor[group_name]["loc"] = np.full(
                    shape=(n, self.n_total_factors), fill_value=self._init_factors, dtype=np.float32
                )
                init_tensor[group_name]["scale"] = np.full(
                    shape=(n, self.n_total_factors), fill_value=self._init_scale, dtype=np.float32
                )
            return init_tensor
        match self._init_factors:
            case "random":
                for group_name, n in data.n_samples.items():
                    init_tensor[group_name]["loc"] = np.random.uniform(size=(n, self.n_total_factors))
            case "orthogonal":
                for group_name, n in data.n_samples.items():
                    # Compute PCA of random vectors
                    pca = PCA(n_components=self.n_total_factors, whiten=True)
                    pca.fit(stats.norm.rvs(loc=0, scale=1, size=(n, self.n_total_factors)).T)
                    init_tensor[group_name]["loc"] = pca.components_.T
            case "pca" | "nmf" as init:
                if init == "pca":
                    initializer = PCA(n_components=self.n_total_factors, whiten=True)
                elif init == "nmf":
                    initializer = NMF(n_components=self.n_total_factors, max_iter=1000)

                inits = data.apply(
                    self._init_factor_group, by_view=False, impute_missings=impute_missings, initializer=initializer
                )
                for group_name, init in inits.items():
                    init_tensor[group_name]["loc"] = init
            case _:
                raise ValueError(
                    f"Initialization method '{self._init_factors}' not found. Please choose from 'random', 'orthogonal', 'PCA', or 'NMF'."
                )

        for group_name, n in data.n_samples.items():
            # scale factor values from -1 to 1 (per factor)
            q = init_tensor[group_name]["loc"]

            if q.shape[0] > 1:  # min and max are not defined for dimensions of size 1
                q = 2.0 * (q - np.min(q, axis=0)) / (np.max(q, axis=0) - np.min(q, axis=0)) - 1
            elif n > 0:
                q = 2.0 * (q - np.min(q)) / (np.max(q) - np.min(q)) - 1

            # Add artifical dimension at dimension -2 for broadcasting
            init_tensor[group_name]["loc"] = q.astype(np.float32, copy=False)
            init_tensor[group_name]["scale"] = np.full(
                shape=(n, self.n_total_factors), fill_value=self._init_scale, dtype=np.float32
            )

        return init_tensor

    def on_train_start(self, data: MofaFlexDataset, sample_plate_dim: int, feature_plate_dim: int):
        if self.n_guided_factors > 0:
            self._init_guiding_vars(data)
        self._factor_names = np.asarray(self._factor_names)
        self._factor_order = np.arange(self.n_total_factors)

        if self._init_factors is not None:
            # need to call contiguous() here, otherwise we get a warning from PyTorch:
            # grad and param do not obey the gradient layout contract
            factors_init_tensor = {
                name: {sname: torch.as_tensor(sval).contiguous() for sname, sval in val.items()}
                for name, val in self._initialize_factors(data).items()
            }
        else:
            factors_init_tensor = None

        for prior in self._factor_priors:
            prior.on_train_start(self.n_total_factors, data.n_samples, factors_init_tensor)
        for prior in self._weight_priors:
            prior.on_train_start(self.n_total_factors, data.n_features)

        for dsets in self._weight_dsets.values():
            for view_name, dset in dsets.items():
                if data.cast_to is not None:
                    dset = dset.astype(data.cast_to, copy=False)
                dsets[view_name] = torch.as_tensor(dset)

    def on_train_epoch_start(self, epoch: int):
        for prior in chain(self._factor_priors, self._weight_priors):
            prior.on_train_epoch_start(epoch)

    def on_train_epoch_end(self, epoch: int):
        for prior in chain(self._factor_priors, self._weight_priors):
            prior.on_train_epoch_end(epoch)

    def on_train_end(self, data: MofaFlexDataset, batch_size: int):
        with torch.inference_mode():
            for priors, nonnegative, names, attrname in zip(
                (self._factor_priors, self._weight_priors),
                (self._nonnegative_factors, self._nonnegative_weights),
                (data.group_names, data.view_names),
                ("_factors", "_weights"),
                strict=True,
            ):
                res = MeanStd({}, {})
                for prior in priors:
                    for lsidx, vals in enumerate(prior.posterior):
                        res[lsidx].update(vals)

                for name in names:
                    if nonnegative[name]:
                        res.mean[name] = self._pos_transform(res.mean[name])
                    res.mean[name] = res.mean[name].cpu().numpy()
                    with suppress(KeyError):
                        res.std[name] = res.std[name].cpu().numpy()
                setattr(self, attrname, res)

        for prior in self._factor_priors:
            prior.on_train_end(
                data, self._factor_names, data.sample_names, self._factors, self._nonnegative_factors, batch_size
            )
        for prior in self._weight_priors:
            prior.on_train_end(
                data, self._factor_names, data.feature_names, self._weights, self._nonnegative_weights, batch_size
            )

        self._device = default_torch_device()
        self._init_api()

    def _get_plates(self, id: str):
        if self.n_guided_factors > 0:
            guiding_var_plate = pyro.plate(
                f"{id}_plate_guiding_vars", 1, subsample=torch.arange(1), dim=self._feature_plate_dim
            )
            guiding_var_coefficients_plate = pyro.plate(f"{id}_plate_guiding_vars_coefficients", 2, dim=-1)
            guiding_var_categories_plates = {}
            for guiding_var_name in self._guiding_vars_names:
                guiding_var_categories_plates[guiding_var_name] = pyro.plate(
                    f"{id}_plate_guiding_var_categories_{guiding_var_name}",
                    self._guiding_vars_weights_dims[guiding_var_name],
                    dim=-2,
                )
        else:
            guiding_var_plate = guiding_var_coefficients_plate = guiding_var_categories_plates = None

        factors_plate = pyro.plate(f"{id}_plate_factors", self.n_total_factors, dim=-1)

        return guiding_var_plate, guiding_var_coefficients_plate, guiding_var_categories_plates, factors_plate

    def _model_guiding_vars_weights_normal(
        self, id, guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates, **kwargs
    ):
        weights_dim = self._guiding_vars_weights_dims[guiding_var_name]
        with guiding_var_categories_plates[guiding_var_name], guiding_var_coefficients_plate:
            return pyro.sample(
                f"{id}_guiding_vars_w_{guiding_var_name}",
                dist.Normal(torch.zeros(weights_dim, 2), torch.ones(weights_dim, 2)),  # (categories, intercept & slope)
            )

    def _guide_guiding_vars_weights_normal(
        self, id, guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates, **kwargs
    ):
        with guiding_var_categories_plates[guiding_var_name], guiding_var_coefficients_plate:
            return pyro.sample(
                f"{id}_guiding_vars_w_{guiding_var_name}",
                dist.Normal(self._guiding_locs[guiding_var_name], self._guiding_scales[guiding_var_name]),
            )

    @pyro_method
    def model(
        self,
        id: str,
        sample_plates,
        feature_plates,
        nonmissing_samples,
        nonmissing_features,
        guiding_vars=None,
        **kwargs,
    ):
        guiding_var_plate, guiding_var_coefficients_plate, guiding_var_categories_plates, factor_plate = (
            self._get_plates(id)
        )

        factors = {}
        with change_pyro_plate_dim(sample_plates.values(), -2):
            for i, prior in enumerate(self._factor_priors):
                factors.update(prior.model(f"{id}_factor_{i}", factor_plate, sample_plates, **kwargs))

        for group_name, group_factors in factors.items():
            if self._nonnegative_factors[group_name]:
                factors[group_name] = self._pos_transform(group_factors)

        weights = {}
        with change_pyro_plate_dim(feature_plates.values(), -2):
            for i, prior in enumerate(self._weight_priors):
                weights.update(prior.model(f"{id}_weight_{i}", factor_plate, feature_plates, **self._weight_dsets))

        for view_name, view_weights in weights.items():
            if self._nonnegative_weights[view_name]:
                weights[view_name] = self._pos_transform(view_weights)

        estimates = {}
        for group_name, gnonmissing_samples in nonmissing_samples.items():
            gestimates = {}
            gnonmissing_features = nonmissing_features[group_name]
            for view_name, vnonmissing_samples in gnonmissing_samples.items():
                vnonmissing_features = gnonmissing_features[view_name]

                z = factors[group_name][..., vnonmissing_samples, :]
                w = weights[view_name][..., vnonmissing_features, :]

                gestimates[view_name] = z @ w.mT
            estimates[group_name] = gestimates

        for guiding_var_name, guiding_var_factor_idx in zip(
            self._guiding_vars_names, self._guiding_vars_factors, strict=True
        ):
            w_guiding = self._model_guiding_vars_weights_normal(
                id, guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates
            )

            for group_name, guiding_var in guiding_vars[guiding_var_name].items():
                z_guiding = factors[group_name][..., :, guiding_var_factor_idx, None]

                # (1, n_cats) + (1, n_cats) * (n_samples, 1)
                loc = w_guiding[..., None, :, 0] + w_guiding[..., None, :, 1] * z_guiding  # (n_samples, n_cats)

                if self._guiding_vars_n_categories[guiding_var_name] > 0:
                    loc = loc.unsqueeze(
                        self._feature_plate_dim - 1
                    )  # Categorical likelihood needs separate dimension for categories

                self._pyro_guiding_vars_likelihoods[guiding_var_name].model(
                    f"{id}_{guiding_var_name}",
                    data=guiding_var,
                    estimate=loc,
                    group_name=group_name,
                    scale=self._guiding_vars_scales[guiding_var_name],
                    sample_plate=sample_plates[group_name],
                    feature_plate=guiding_var_plate,
                    nonmissing_samples=slice(None),
                    nonmissing_features=slice(None),
                )
        return estimates

    @pyro_method
    def guide(
        self,
        id: str,
        sample_plates,
        feature_plates,
        nonmissing_samples,
        nonmissing_features,
        guiding_vars=None,
        **kwargs,
    ):
        (guiding_var_plate, guiding_var_coefficients_plate, guiding_var_categories_plates, factor_plate) = (
            self._get_plates(id)
        )

        with change_pyro_plate_dim(sample_plates.values(), -2):
            for i, prior in enumerate(self._factor_priors):
                prior.guide(f"{id}_factor_{i}", factor_plate, sample_plates, **kwargs)

        with change_pyro_plate_dim(feature_plates.values(), -2):
            for i, prior in enumerate(self._weight_priors):
                prior.guide(f"{id}_weight_{i}", factor_plate, feature_plates, **self._weight_dsets)

        if self.n_guided_factors > 0:
            for guiding_var_name, guiding_var in guiding_vars.items():
                self._guide_guiding_vars_weights_normal(
                    id, guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates
                )
                for group_name in guiding_var.keys():
                    self._pyro_guiding_vars_likelihoods[guiding_var_name].guide(
                        f"{id}_{guiding_var_name}", group_name, sample_plates[group_name], guiding_var_plate
                    )

    @property
    def learning_rate_multipliers(self) -> Iterable[tuple[str, float]]:
        for i, prior in enumerate(self._factor_priors):
            yield from ((f"_factor_priors.{i}.{pname}", mod) for pname, mod in prior.learning_rate_multipliers)
        for i, prior in enumerate(self._weight_priors):
            yield from ((f"_weight_priors.{i}.{pname}", mod) for pname, mod in prior.learning_rate_multipliers)

    @property
    def nonnegative(self):
        return {
            group_name: {view_name: gfactors & vweights for view_name, vweights in self._nonnegative_weights.items()}
            for group_name, gfactors in self._nonnegative_factors.items()
        }

    def predict(
        self,
        group_name: str,
        view_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
        idx_cartesian_product: bool = True,
    ) -> NDArray[np.floating]:
        if idx_cartesian_product:
            return (
                self._get_postprocessed_factors("mean", group_name)[sample_idx]
                @ self._get_postprocessed_weights("mean", view_name)[feature_idx].T
            )
        else:
            return (
                self._get_postprocessed_factors("mean", group_name)[sample_idx]
                * self._get_postprocessed_weights("mean", view_name)[feature_idx]
            ).sum(axis=1)

    def prediction_components(
        self,
        group_name: str,
        view_name: str,
        sample_idx: NDArray[int] | slice = slice(None),
        feature_idx: NDArray[int] | slice = slice(None),
        idx_cartesian_product: bool = True,
    ) -> Iterable[tuple[str, NDArray[np.floating]]]:
        if idx_cartesian_product:
            yield from (
                (
                    factor_name,
                    self._get_postprocessed_factors("mean", group_name)[sample_idx, factor_idx, None]
                    @ self._get_postprocessed_weights("mean", view_name)[None, feature_idx, factor_idx],
                )
                for factor_idx, factor_name in enumerate(self.factor_names)
            )
        else:
            yield from (
                (
                    factor_name,
                    self._get_postprocessed_factors("mean", group_name)[sample_idx, factor_idx]
                    * self._get_postprocessed_weights("mean", view_name)[feature_idx, factor_idx],
                )
                for factor_idx, factor_name in enumerate(self.factor_names)
            )

    def _save(self) -> dict[str, Any]:
        return {
            "factor_priors": {str(i): prior.save() for i, prior in enumerate(self._factor_priors)},
            "weight_priors": {str(i): prior.save() for i, prior in enumerate(self._weight_priors)},
        }

    def _load(
        self,
        state: Mapping[str, Any],
        sample_names: Mapping[str, NDArray[str]],
        feature_names: Mapping[str, NDArray[str]],
        n_samples: Mapping[str, int],
        n_features: Mapping[str, int],
        map_location=None,
        **kwargs,
    ):
        self._sample_names = sample_names
        self._feature_names = feature_names
        self._factor_priors = PyroModuleList(
            Prior.load(pstate, map_location=map_location, n_factors=self.n_total_factors, n_nonfactors=n_samples)
            for pstate in state["factor_priors"].values()
        )
        self._weight_priors = PyroModuleList(
            Prior.load(pstate, map_location=map_location, n_factors=self.n_total_factors, n_nonfactors=n_features)
            for pstate in state["weight_priors"].values()
        )

        self._prior_api_properties = {}
        self._device = map_location
        self._init_api()

    def _get_postprocessed_factors(
        self, moment: Literal["mean", "std"] = "mean", group_name: str | None = None, **kwargs
    ) -> dict[str, np.ndarray]:
        factors = {}
        for prior in self._factor_priors:
            if (
                postprocessed := prior.postprocess_results(self._factors, moment=moment, name=group_name, **kwargs)
            ) is not None:
                if group_name is not None:
                    return postprocessed
                else:
                    factors.update(postprocessed)
        return factors

    @Term._api
    def get_factors(  # noqa: D417
        self, moment: Literal["mean", "std"] = "mean", ordered: bool = False, **kwargs
    ) -> dict[str, pd.DataFrame | AnnData]:
        """Get the factor matrices Z for each group.

        Args:
            moment: Which moment of the posterior distribution to return.
            ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        factors = self._get_postprocessed_factors(moment, **kwargs)
        return self._results_to_df(factors, axis=0, ordered=ordered)

    def _get_postprocessed_weights(
        self, moment: Literal["mean", "std"] = "mean", view_name: str | None = None, **kwargs
    ) -> dict[str, np.ndarray]:
        weights = {}
        for prior in self._weight_priors:
            if (
                postprocessed := prior.postprocess_results(self._weights, moment=moment, name=view_name, **kwargs)
            ) is not None:
                if view_name is not None:
                    return postprocessed
                else:
                    weights.update(postprocessed)
        return weights

    @Term._api
    def get_weights(  # noqa: D417
        self, moment: Literal["mean", "std"] = "mean", ordered: bool = False, **kwargs
    ) -> dict[str, pd.DataFrame]:
        """Get the weight matrices W for each view.

        Args:
            moment: Which moment of the posterior distribution to return.
            ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        weights = self._get_postprocessed_weights(moment, **kwargs)
        return self._results_to_df(weights, axis=1, ordered=ordered)

    def _make_factor_names_unique(self):
        namescounter = Counter(self._factor_names)
        seen = defaultdict(lambda: 1)
        for i, fname in enumerate(self._factor_names):
            if namescounter[fname] > 1:
                self._factor_names[i] = f"{fname}-{seen[fname]}"
                seen[fname] += 1


# init API for docs
def _init_api():

    def raise_(exc):
        raise exc

    def make_dummy_function(name: str, prior: str, is_property: bool):
        if is_property:
            return lambda self: raise_(
                AttributeError(
                    f"The '{name}' property is only available when using the '{prior}' prior.", obj=self, name=name
                )
            )
        else:
            return lambda self, *args, **kwargs: raise_(
                AttributeError(
                    f"The '{name}' method is only available when using the '{prior}' prior.", obj=self, name=name
                )
            )

    apinames: dict[tuple[int, str, str], str] = {}

    getters = MofaFlex.get_factors, MofaFlex.get_weights
    getter_sigs = tuple(inspect.signature(getter) for getter in getters)
    getter_params = tuple([param for param in sig.parameters.values() if param.name != "kwargs"] for sig in getter_sigs)
    getter_ignored_params = (
        {"self", "results", "moment", "name", "kwargs"},
        {"self", "results", "moment", "name", "kwargs"},
    )
    getter_annots = tuple(getter.__annotations__ for getter in getters)

    for axis, axisname, priors in (
        (0, "factor", Prior.known_priors("factors")),
        (1, "weight", Prior.known_priors("weights")),
    ):
        namescount = Counter(api.name for api in chain(*(x.api() for x in priors.values())))
        duplicates = {k for k, v in namescount.items() if v > 1}

        for prior, priorcls in priors.items():
            for api in priorcls.api():
                name = api.name if api.name not in duplicates else f"{api.name}_{prior}"
                name = name.replace("a̲x̲i̲s̲", axisname)
                if api.type == APIType.property and api.has_factors:
                    name = f"get_{name}"
                apinames[(axis, prior, api.name)] = name

                if api.type == APIType.property and not api.has_factors:
                    attr = property(make_dummy_function(name, prior, True))
                    attr.__doc__ = getattr(priorcls, api.name).__doc__
                    setattr(MofaFlex, name, attr)
                    MofaFlex._api(name, hidden=True)
                    continue

                func = getattr(priorcls, api.name)
                if api.type == APIType.property:
                    func = func.fget
                doc = func.__doc__
                sig = inspect.signature(func)
                params = list(sig.parameters.values())
                annots = func.__annotations__.copy()
                wrapperfunc = make_dummy_function(name, prior, False)
                wrapperfunc.__doc__ = doc
                if not api.has_factors:
                    wrapperfunc.__signature__ = sig
                else:
                    params.append(
                        inspect.Parameter(
                            "ordered", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=False, annotation=bool
                        )
                    )
                    annots["ordered"] = bool
                    annots["return"] = Mapping[str, pd.DataFrame]
                    wrapperfunc.__signature__ = sig.replace(parameters=params)
                    wrapperfunc.__annotations__ = annots
                    wrapperfunc.__qualname__ = f"{MofaFlex.__qualname__}.{name}"
                    wrapperfunc.__name__ = name

                setattr(MofaFlex, name, wrapperfunc)
                MofaFlex._api(name, hidden=True)

            postprocess_method = priorcls.postprocess_results
            params = [
                param
                for param in inspect.signature(postprocess_method).parameters.values()
                if param.name not in getter_ignored_params[axis]
            ]
            if len(params) > 0:
                for param in params:
                    if param.name not in getter_sigs[axis].parameters:
                        getter_params[axis].append(param)
                        getter_annots[axis][param.name] = param.annotation
                    else:
                        getter_annots[axis][param.name] |= param.annotation
                    getter_ignored_params[axis].add(param.name)

    # can't move this inside the loop due to Python's late binding closures
    getter_wrappers = (
        lambda self, *args, **kwargs: getters[0](self, *args, **kwargs),
        lambda self, *args, **kwargs: getters[1](self, *args, **kwargs),
    )
    for axis, (method, wrapper) in enumerate(zip(getters, getter_wrappers, strict=True)):
        wrapper.__signature__ = getter_sigs[axis].replace(parameters=getter_params[axis])
        wrapper.__annotations__ = getter_annots[axis]
        wrapper.__doc__ = getters[axis].__doc__
        wrapper.__qualname__ = method.__qualname__
        wrapper.__name__ = method.__name__
        setattr(MofaFlex, method.__name__, wrapper)

    return apinames


_apinames = _init_api()
