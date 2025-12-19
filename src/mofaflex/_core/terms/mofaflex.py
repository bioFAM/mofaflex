import inspect
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from functools import update_wrapper
from itertools import chain
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

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

from ..datasets import CovariatesDataset, MofaFlexDataset, StackDataset
from ..likelihoods.pyro import PyroLikelihood
from ..priors import API, APIType, FactorPriorType, Prior, WeightPriorType
from ..utils import MeanStd, PyroModuleDict, PyroParameterDict
from .base import Term

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..api.priors import Prior as APIPrior


class _PriorApiProperty(NamedTuple):
    obj: Prior
    attr: str


class MofaFlex(Term):
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
        factor_prior: Mapping[str | Sequence[str], FactorPriorType | "APIPrior"]
        | FactorPriorType
        | "APIPrior" = "Normal",
        weight_prior: Mapping[str | Sequence[str], WeightPriorType | "APIPrior"]
        | WeightPriorType
        | "APIPrior" = "Normal",
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
            if ordered:
                factor_order = self.factor_order[factors_subset]
                factor_order = np.argsort(np.argsort(factor_order))
                res = res[:, factor_order]
            ret[name] = pd.DataFrame(
                res, index=self._sample_names[name] if axis == 0 else self._feature_names[name], columns=factor_names
            )
        return ret

    def _wrap_api_method(self, axis: Literal[0, 1], prior: Prior, api: API):
        def wrapper_func(self, *args, **kwargs):
            with torch.device(self._train_opts.device):
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

    def _init_api(self):
        for axis, priors in ((0, self._factor_priors), (1, self._weight_priors)):
            for prior in priors:
                for api in prior.api():
                    name = _apinames[(axis, prior.__class__.__name__, api.name)]
                    if api.type == APIType.property and not api.has_factors:
                        self._prior_api_properties[name] = _PriorApiProperty(prior, api.name)
                        continue
                    wrapped = self._wrap_api_method(axis, prior, api)
                    dummy = getattr(self.__class__, name)
                    update_wrapper(wrapped, dummy)
                    setattr(self, name, wrapped.__get__(self))

    def __getattribute__(self, name):
        try:
            prop = super().__getattribute__("_prior_api_properties")[name]
            return getattr(prop.obj, prop.attr)
        except (KeyError, AttributeError):
            return super().__getattribute__(name)

    def __dir__(self):
        return chain(super().__dir__(), self._prior_api_properties.keys())

    @property
    def n_guided_factors(self) -> int:
        return len(self._guiding_vars_names)

    @property
    def _guiding_vars_factors(self) -> range:
        return range(self.n_total_factors - self.n_guided_factors, self.n_total_factors)

    @property
    def n_factors(self) -> int:
        return self._n_factors

    @property
    def n_total_factors(self) -> int:
        return len(self._factor_names)

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

    @property
    def factor_order(self) -> NDArray[int]:
        return self._factor_order

    @factor_order.setter
    def factor_order(self, order: NDArray[int]):
        self.component_order = order

    def _init(self, data: MofaFlexDataset):
        self._sample_names = data.sample_names
        self._feature_names = data.feature_names
        from ..api.priors import Prior as APIPrior

        self._pos_transform = torch.nn.ReLU()
        for axis, (priorattr, names) in enumerate(
            zip(("_factor_priors", "_weight_priors"), (data.group_names, data.view_names), strict=True)
        ):
            priors = getattr(self, priorattr)
            if isinstance(priors, str):
                priors = [Prior(priors, axis=axis, names=names)]
            elif isinstance(priors, APIPrior):
                priors = [priors(axis=axis, names=names)]
            else:
                prior_groups = defaultdict(list)
                for group_name, prior in priors.items():
                    if isinstance(group_name, str):
                        prior_groups[prior].append(group_name)
                    else:
                        prior_groups[prior].extend(group_name)
                priors = []
                for priorname, names in prior_groups.items():
                    if isinstance(priorname, str):
                        prior = Prior(priorname, axis=axis, names=names)
                    else:
                        prior = priorname(axis=axis, names=names)
                    priors.append(prior)
            setattr(self, priorattr, PyroModuleList(priors))

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
            self._guiding_vars_names = self._guiding_vars_obs_keys.keys()
        else:
            self._guiding_vars_names = []

        if self.n_guided_factors > 0:
            if not isinstance(self._guiding_vars_scales, dict):
                self._guiding_vars_scales = dict.fromkeys(self._guiding_vars_names, self._guiding_vars_scales)

            total_n_features = 0.1 * data.n_features_total
            self._guiding_vars_scales = {
                name: scale * total_n_features for name, scale in self._guiding_vars_scales.items()
            }

            self._pyro_guiding_vars_likelihoods = PyroModuleDict(
                {
                    guiding_var_name: PyroLikelihood(
                        self._guiding_vars_likelihoods[guiding_var_name],
                        view_name=guiding_var_name,
                        sample_dim=self._sample_plate_dim,
                        feature_dim=self._feature_plate_dim,
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
                    torch.full([weights_dim, 2]), constraint=constraints.real
                )
                self._guiding_scales[guiding_var_name] = PyroParam(
                    torch.full([weights_dim, 2]), constraint=constraints.softplus_positive
                )

    _sample_plate_dim = -2
    _feature_plate_dim = -1
    _factor_plate_dim = -3

    def get_datasets(self, data: MofaFlexDataset) -> dict[str, CovariatesDataset]:
        self._init(data)

        ret = {}
        for prior in chain(self._factor_priors, self._weight_priors):
            if priordsets := prior.get_datasets(data):
                ret.update(priordsets)

        if self.n_guided_factors > 0:
            guiding_vars = {}
            for guiding_var_name, obs_key in self._guiding_vars_obs_keys.items():
                guiding_vars[guiding_var_name] = CovariatesDataset(data, obs_key=obs_key)
            ret["guiding_vars"] = guiding_vars = StackDataset(**guiding_vars)

            for guiding_var_name, guiding_var_likelihood in self._guiding_vars_likelihoods.items():
                if guiding_var_likelihood == "Categorical":
                    guiding_vars_categories = set()
                    # find number of unique categories across groups
                    for group_name in data.group_names:
                        guiding_vars_categories.update(
                            guiding_vars.datasets[guiding_var_name].covariates[group_name].iloc[:, 0].to_list()
                        )
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
                raise ValueError("Data has missing values. Please impute missings or set `impute_missings=True`.")
        return initializer.fit_transform(arr)

    def _initialize_factors(self, data, impute_missings=True):
        init_tensor = defaultdict(dict)
        _logger.info(f"Initializing factors using `{self._init_factors}` method...")

        if not isinstance(self._init_factors, str):
            for group_name, n in data.n_samples.items():
                init_tensor[group_name]["loc"] = np.full(
                    shape=(n, self.n_total_factors), fill_value=self._init_factors, dtype=np.float32
                ).T[..., None]
                init_tensor[group_name]["scale"] = np.full(
                    shape=(n, self.n_total_factors), fill_value=self._init_scale, dtype=np.float32
                ).T[..., None]
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
                    f"Initialization method `{self._init_factors}` not found. Please choose from `random`, `orthogonal`, `PCA`, or `NMF`."
                )

        for group_name, n in data.n_samples.items():
            # scale factor values from -1 to 1 (per factor)
            q = init_tensor[group_name]["loc"]

            if q.shape[0] > 1:  # min and max are not defined for dimensions of size 1
                q = 2.0 * (q - np.min(q, axis=0)) / (np.max(q, axis=0) - np.min(q, axis=0)) - 1
            elif n > 0:
                q = 2.0 * (q - np.min(q)) / (np.max(q) - np.min(q)) - 1

            # Add artifical dimension at dimension -2 for broadcasting
            init_tensor[group_name]["loc"] = q.T[..., None].astype(np.float32, copy=False)
            init_tensor[group_name]["scale"] = np.full(
                shape=(n, self.n_total_factors), fill_value=self._init_scale, dtype=np.float32
            ).T[..., None]

        return init_tensor

    def on_train_start(self, data: MofaFlexDataset):
        for prior in chain(self._factor_priors, self._weight_priors):
            self._factor_names = prior.adjust_factors(self._factor_names)

        if self.n_guided_factors > 0:
            self._factor_names = np.concatenate((self._factor_names, self._guiding_vars_names))
        else:
            self._factor_names = np.asarray(self._factor_names)
        self._factor_order = np.arange(self._n_factors)

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
            prior.on_train_start(
                self._factor_plate_dim,
                self._sample_plate_dim,
                self.n_total_factors,
                data.n_samples,
                factors_init_tensor,
            )
        for prior in self._weight_priors:
            prior.on_train_start(self._factor_plate_dim, self._feature_plate_dim, self.n_total_factors, data.n_features)

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
                    res.mean[name] = res.mean[name].cpu().numpy().T
                    res.std[name] = res.std[name].cpu().numpy().T
                setattr(self, attrname, res)

        for prior in self._factor_priors:
            prior.on_train_end(
                data, self._factor_names, data.sample_names, self._factors, self._nonnegative_factors, batch_size
            )
        for prior in self._weight_priors:
            prior.on_train_end(
                data, self._factor_names, data.feature_names, self._weights, self._nonnegative_weights, batch_size
            )

        self._init_api()

    def _get_plates(self):
        if self.n_guided_factors > 0:
            guiding_var_plate = pyro.plate(
                "plate_guiding_vars", 1, subsample=torch.arange(1), dim=self._feature_plate_dim
            )
            guiding_var_coefficients_plate = pyro.plate("plate_guiding_vars_coefficients", 2, dim=-1)
            guiding_var_categories_plates = {}
            for guiding_var_name in self._guiding_vars_names:
                guiding_var_categories_plates[guiding_var_name] = pyro.plate(
                    f"plate_guiding_var_categories_{guiding_var_name}",
                    self._guiding_vars_weights_dims[guiding_var_name],
                    dim=-2,
                )
        else:
            guiding_var_plate = guiding_var_coefficients_plate = guiding_var_categories_plates = None

        factors_plate = pyro.plate("plate_factors", self.n_total_factors, dim=-3)

        return guiding_var_plate, guiding_var_coefficients_plate, guiding_var_categories_plates, factors_plate

    def _model_guiding_vars_weights_normal(
        self, guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates, **kwargs
    ):
        weights_dim = self._guiding_vars_weights_dims[guiding_var_name]
        with guiding_var_categories_plates[guiding_var_name], guiding_var_coefficients_plate:
            return pyro.sample(
                f"guiding_vars_w_{guiding_var_name}",
                dist.Normal(
                    torch.zeros(weights_dim, 2), torch.ones(weights_dim, 2)
                ),  # .to_event(2)  # (categories, intercept & slope)
            )

    def _guide_guiding_vars_weights_normal(
        self, guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates, **kwargs
    ):
        with guiding_var_categories_plates[guiding_var_name], guiding_var_coefficients_plate:
            return pyro.sample(
                f"guiding_vars_w_{guiding_var_name}",
                dist.Normal(self._guiding_locs[guiding_var_name], self._guiding_scales[guiding_var_name]),
            )

    @pyro_method
    def model(
        self, sample_plates, feature_plates, nonmissing_samples, nonmissing_features, guiding_vars=None, **kwargs
    ):
        guiding_var_plate, guiding_var_coefficients_plate, guiding_var_categories_plates, factor_plate = (
            self._get_plates()
        )

        factors = {}
        for prior in self._factor_priors:
            factors.update(prior.model(factor_plate, sample_plates, **kwargs))

        for group_name, group_factors in factors.items():
            if self._nonnegative_factors[group_name]:
                factors[group_name] = self._pos_transform(group_factors)

        weights = {}
        for prior in self._weight_priors:
            weights.update(prior.model(factor_plate, feature_plates))

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
                w = weights[view_name][..., vnonmissing_features]

                gestimates[view_name] = torch.einsum("...ijk,...ikl->...kjl", z, w)
            estimates[group_name] = gestimates

        for guiding_var_name, guiding_var_factor_idx in zip(
            self._guiding_vars_names, self._guiding_vars_factors, strict=True
        ):
            w_guiding = self._model_guiding_vars_weights_normal(
                guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates
            )

            for group_name, guiding_var in guiding_vars[guiding_var_name].items():
                z_guiding = factors[group_name].select(factor_plate.dim, guiding_var_factor_idx)

                # (1, n_cats) + (1, n_cats) * (n_samples, 1)
                loc = (
                    torch.atleast_2d(w_guiding[..., 0]) + torch.atleast_2d(w_guiding[..., 1]) * z_guiding
                )  # (n_samples, n_cats)

                if self._guiding_vars_n_categories[guiding_var_name] > 0:
                    loc = loc.unsqueeze(
                        self._feature_plate_dim - 1
                    )  # Categorical likelihood needs separate dimension for categories

                self._pyro_guiding_vars_likelihoods[guiding_var_name].model(
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
        self, sample_plates, feature_plates, nonmissing_samples, nonmissing_features, guiding_vars=None, **kwargs
    ):
        (guiding_var_plate, guiding_var_coefficients_plate, guiding_var_categories_plates, factor_plate) = (
            self._get_plates()
        )

        for prior in self._factor_priors:
            prior.guide(factor_plate, sample_plates, **kwargs)

        for prior in self._weight_priors:
            prior.guide(factor_plate, feature_plates)

        if self.n_guided_factors > 0:
            for guiding_var_name, guiding_var in guiding_vars.items():
                self._guide_guiding_vars_weights_normal(
                    guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates
                )
                for group_name in guiding_var.keys():
                    self._pyro_guiding_vars_likelihoods[guiding_var_name].guide(
                        group_name, sample_plates[group_name], guiding_var_plate
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

    def predict(self, group_name: str, view_name: str, subsample_idx: NDArray[int] | slice = slice(None)):
        return self._factors.mean[group_name][subsample_idx] @ self._weights.mean[view_name].T

    def prediction_components(
        self, group_name: str, view_name: str, subset_idx: NDArray[int] | slice = slice(None)
    ) -> Iterable[tuple[str, NDArray[np.floating]]]:
        yield from (
            (
                factor_name,
                self._factors.mean[group_name][subset_idx, factor_idx, None]
                @ self._weights.mean[view_name][None, :, factor_idx],
            )
            for factor_idx, factor_name in enumerate(self.factor_names)
        )

    def _save(self) -> dict[str, Any]:
        return {
            "factor_priors": {str(i): prior.save() for i, prior in enumerate(self._factor_priors)},
            "weight_priors": {str(i): prior.save() for i, prior in enumerate(self._weight_priors)},
        }

    def _load(
        self, state: dict[str, Any], n_samples: dict[str, int], n_features: dict[str, int], map_location=None, **kwargs
    ):
        self._factor_priors = PyroModuleList(
            Prior.load(
                pstate,
                n_samples=n_samples,
                n_features=n_features,
                map_location=map_location,
                n_factors=self.n_total_factors,
                n_nonfactors=n_samples,
            )
            for pstate in state["factor_priors"].values()
        )
        self._weight_priors = PyroModuleList(
            Prior.load(
                pstate,
                n_samples=n_samples,
                n_features=n_features,
                map_location=map_location,
                n_factors=self.n_total_factors,
                n_nonfactors=n_features,
            )
            for pstate in state["weight_priors"].values()
        )

        self._prior_api_properties = {}
        self._init_api()

    def _get_postprocessed_factors(self, moment: Literal["mean", "std"] = "mean", **kwargs) -> dict[str, np.ndarray]:
        factors = {}
        for prior in self._factor_priors:
            factors.update(prior.postprocess_results(self._factors, moment=moment, **kwargs))
        return factors

    def get_factors(  # noqa: D417
        self,
        moment: Literal["mean", "std"] = "mean",
        ordered: bool = False,
        return_type: Literal["pandas", "anndata"] = "pandas",
        **kwargs,
    ) -> dict[str, pd.DataFrame | AnnData]:
        """Get the factor matrices Z for each group.

        Args:
            moment: Which moment of the posterior distribution to return.
            ordered: Whether to return the factors ordered by explained variance (highest to lowest).
            return_type: Format of the returned object.
        """
        factors = self._get_postprocessed_factors(moment, **kwargs)
        factors = self._results_to_df(factors, axis=0, ordered=ordered)

        if return_type == "anndata":
            for group_name, group_factors in factors.items():
                group_adata = AnnData(group_factors)
                group_adata.obs = pd.concat(self._metadata[group_name].values(), axis=1)
                group_adata.obs = group_adata.obs.loc[:, ~group_adata.obs.columns.duplicated()]
                factors[group_name] = group_adata

        return factors

    def _get_postprocessed_weights(self, moment: Literal["mean", "std"] = "mean", **kwargs) -> dict[str, np.ndarray]:
        weights = {}
        for prior in self._weight_priors:
            weights.update(prior.postprocess_results(self._weights, moment=moment, **kwargs))
        return weights

    def get_weights(  # noqa: D417
        self, moment: Literal["mean", "std"] = "mean", ordered: bool = False, **kwargs
    ) -> dict[str, pd.DataFrame]:
        """Get the weight matrices W for each view.

        Args:
            return_type: Format of the returned object.
            moment: Which moment of the posterior distribution to return.
            ordered: Whether to return the factors ordered by explained variance (highest to lowest).
        """
        weights = self._get_postprocessed_weights(moment, **kwargs)
        weights = self._results_to_df(weights, axis=1, ordered=ordered)

        return weights


# init API for docs
def _init_api():
    def raise_(exc):
        raise exc

    def get_line_indentation(line: str):
        for i, s in enumerate(line):
            if not s.isspace():
                return i
        return np.inf

    def get_indentation(docstring: str):
        if not docstring:
            return 0
        lines = docstring.expandtabs(4).splitlines()
        min_indent = np.inf
        for line in lines[1:]:
            min_indent = min(min_indent, get_line_indentation(line))
        return min_indent if np.isfinite(min_indent) else 0

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
    getter_annots = tuple(getter.__annotations__ for getter in getters)
    getter_docs = [getter.__doc__ for getter in getters]
    getter_indents = [" " * get_indentation(doc) for doc in getter_docs]

    for axis, axisname, priors in (
        (0, "factor", Prior.known_priors("factors")),
        (1, "weight", Prior.known_priors("weights")),
    ):
        namescount = Counter()
        for api in chain(*(x.api() for x in priors.values())):
            namescount[api.name] += 1
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
                    attr.__doc__ = (
                        getattr(priorcls, api.name).__doc__ + "\n\n.. important::\n"
                        f"   This property is only available when using the {prior} prior."
                    )
                    setattr(MofaFlex, name, attr)
                    continue

                func = getattr(priorcls, api.name)
                if api.type == APIType.property:
                    func = func.fget
                doc = func.__doc__
                sig = inspect.signature(func)
                params = list(sig.parameters.values())
                annots = func.__annotations__.copy()
                wrapperfunc = make_dummy_function(name, prior, False)
                if not api.has_factors:
                    wrapperfunc.__doc__ = doc
                else:
                    if doc is not None:
                        doc += "\n\n"
                    else:
                        doc = ""
                    indent = " " * get_indentation(doc)
                    wrapperfunc.__doc__ = (
                        doc + f"{indent}Args:\n"
                        f"{indent}    ordered: Whether to return the factors ordered by explained variance (highest to lowest).\n\n"
                        f"{indent}.. important::\n"
                        f"{indent}   This method is only available when using the `{prior}` prior."
                    )
                    params.append(
                        inspect.Parameter(
                            "ordered", inspect.Parameter.POSITIONAL_OR_KEYWORD, default=False, annotation=bool
                        )
                    )
                    annots["ordered"] = bool
                    wrapperfunc.__signature__ = sig.replace(parameters=params)
                    wrapperfunc.__annotations__ = annots
                    wrapperfunc.__qualname__ = f"{MofaFlex.__qualname__}.{name}"
                    wrapperfunc.__name__ = name
                setattr(MofaFlex, name, wrapperfunc)

            postprocess_method = priorcls.postprocess_results
            params = [
                param
                for param in inspect.signature(postprocess_method).parameters.values()
                if param.name not in {"self", "results", "moment", "kwargs"}
            ]
            if len(params) > 0:
                getter_params[axis].extend(params)
                for param in params:
                    getter_annots[axis][param.name] = param.annotation
                if doc := postprocess_method.__doc__:
                    docindent = get_indentation(doc)
                    lines = doc.expandtabs(4).splitlines()
                    lines[0] = getter_indents[axis] + "Args:"
                    for i, line in enumerate(lines[1:]):
                        lines[i + 1] = getter_indents[axis] + "    " + line[docindent:]
                    doc = "\n".join(lines)
                    getter_docs[axis] += (
                        "\n"
                        + doc
                        + f"\n{getter_indents[axis]}        .. important::\n{getter_indents[axis]}           This argument is only available when using the `{prior}` prior."
                    )

    # can't move this inside the loop due to Python's late binding closures
    getter_wrappers = (
        lambda self, *args, **kwargs: getters[0](self, *args, **kwargs),
        lambda self, *args, **kwargs: getters[1](self, *args, **kwargs),
    )
    for axis, (method, wrapper) in enumerate(zip(getters, getter_wrappers, strict=True)):
        wrapper.__signature__ = getter_sigs[axis].replace(parameters=getter_params[axis])
        wrapper.__annotations__ = getter_annots[axis]
        wrapper.__doc__ = getter_docs[axis]
        wrapper.__qualname__ = method.__qualname__
        wrapper.__name__ = method.__name__
        setattr(MofaFlex, method.__name__, wrapper)

    return apinames


_apinames = _init_api()
