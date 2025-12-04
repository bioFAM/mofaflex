import inspect
import logging
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import update_wrapper
from itertools import chain
from pathlib import Path
from typing import Literal, NamedTuple, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyro
import torch
from anndata import AnnData
from array_api_compat import array_namespace
from mudata import MuData
from pyro.infer import SVI, TraceMeanField_ELBO
from pyro.optim import ClippedAdam
from scipy import stats
from scipy.sparse import issparse
from sklearn.decomposition import NMF, PCA
from torch.utils.data import DataLoader, default_convert
from torch.utils.data._utils.collate import collate  # this is documented, so presumably part of the public API
from tqdm.auto import tqdm
from tqdm.notebook import tqdm_notebook

from .. import pl
from . import preprocessing
from .datasets import GuidingVarsDataset, MofaFlexBatchSampler, MofaFlexDataset, StackDataset
from .io import MOFACompatOption, load_model, save_model
from .likelihoods import Likelihood, LikelihoodType
from .priors import API, APIType, FactorPriorType, Prior, SmoothOptions, WeightPriorType
from .pyro import MofaFlexModel
from .training import EarlyStopper
from .utils import MeanStd, Options, impute, sample_all_data_as_one_batch

_logger = logging.getLogger(__name__)


class _PriorApiProperty(NamedTuple):
    obj: Prior
    attr: str


@dataclass(kw_only=True)
class DataOptions(Options):
    """Options for the data."""

    group_by: str | Sequence[str] | None = None
    """Columns of `.obs` in :class:`MuData<mudata.MuData>` objects to group data by. Ignored if the input data
    is not a :class:`MuData<mudata.MuData>` object.
    """

    layer: Mapping[str, str | None] | Mapping[str, Mapping[str, str | None]] | str | None = None
    """Which layer to use. If `None`, the `.X` element will be used. If `str`, the same layer will be used for
    all groups and views. If a dict of strings, the keys must correspond to view names and the values to layers.
    If a nested dict, different layers can be used for each combination of group and view. The last format is
    only accepted if the data is a nested dictionary of :class:`AnnData<anndata.AnnData>` objects."""

    scale_per_group: bool = True
    """Scale Normal likelihood data per group, otherwise across all groups."""

    annotations_varm_key: Mapping[str, str] | str | None = None
    """Key of .varm attribute of each AnnData object that contains annotation values."""

    covariates_obs_key: Mapping[str, str] | str | None = None
    """Key of .obs attribute of each :class:`AnnData<anndata.AnnData>` object that contains covariate values."""

    covariates_obsm_key: Mapping[str, str] | str | None = None
    """Key of .obsm attribute of each :class:`AnnData<anndata.AnnData>` object that contains covariate values."""

    guiding_vars_obs_keys: str | Sequence[str] | Mapping[str, Mapping[str, str]] | None = None
    """Keys of .obs attribute of each :class:`AnnData<anndata.AnnData>` object that contains guiding variable values."""

    use_obs: Literal["union", "intersection"] | None = "union"
    """How to align observations across views. Ignored if the data is not a nested dict of :class:`AnnData<anndata.AnnData>` objects."""

    use_var: Literal["union", "intersection"] | None = "union"
    """How to align variables across groups. Ignored if the data is not a nested dict of :class:`AnnData<anndata.AnnData>` objects."""

    subset_var: str | None = "highly_variable"
    """`.var` column with boolean values to select features."""

    plot_data_overview: bool = True
    """Plot data overview."""

    remove_constant_features: bool = True
    """Remove constant features from the data."""


@dataclass(kw_only=True)
class ModelOptions(Options):
    """Options for the model."""

    n_factors: int = 0
    """Number of latent factors."""

    weight_prior: Mapping[str, WeightPriorType] | WeightPriorType = "Normal"
    """Weight priors for each view (if dict) or for all views (if str)."""

    factor_prior: Mapping[str, FactorPriorType] | FactorPriorType = "Normal"
    """Factor priors for each group (if dict) or for all groups (if str)."""

    likelihoods: Mapping[str, LikelihoodType] | LikelihoodType | None = None
    """Data likelihoods for each view (if dict) or for all views (if str). Inferred automatically if None."""

    nonnegative_weights: Mapping[str, bool] | bool = False
    """Non-negativity constraints for weights for each view (if dict) or for all views (if bool)."""

    guiding_vars_likelihoods: Mapping[str, str] | Literal["Normal", "Categorical", "Bernoulli"] | None = "Normal"
    """Likelihood for each guiding variable (if dict) or for all guiding variables (if str)."""

    guiding_vars_scales: Mapping[str, float] | float = 1.0
    """Scale for the likelihood of each guiding variable, to put more or less emphasis on them during training."""

    nonnegative_factors: Mapping[str, bool] | bool = False
    """Non-negativity constraints for factors for each group (if dict) or for all groups (if bool)."""

    annotation_confidence: float = 0.99
    """Confidence in the provided feature annotation. Must be between 0 and 1. Smaller values make the model more likely to
        add features to the annotated pathways during training, while larger values encourage the model to more closely adhere
        to the provided annotations."""

    init_factors: float | Literal["random", "orthogonal", "pca"] = "random"
    """Initialization method for factors."""

    init_scale: float = 0.1
    """Initialization scale of Normal distribution for factors."""


@dataclass(kw_only=True)
class TrainingOptions(Options):
    """Options for training."""

    device: str | torch.device = "cuda"
    """Device to run training on."""

    batch_size: int = 0
    """Batch size."""

    max_epochs: int = 10_000
    """Maximum number of training epochs."""

    n_particles: int = 1
    """Number of particles for ELBO estimation."""

    lr: float = 0.001
    """Learning rate."""

    early_stopper_patience: int = 100
    """Number of steps without relevant improvement to stop training."""

    save_path: Path | str | None = None
    """Path to save model."""

    mofa_compat: MOFACompatOption = False
    """Save model in MOFA2 compatible format. If `True` or `"full"`, will include the data in the file. This
    can result in very large files. `"modelonly"` will save only the trained model."""

    seed: int | None = None
    """Seed for the pseudorandom number generator."""

    num_workers: int = 0
    """Number of data loader workers."""

    pin_memory: bool = False
    """Whether to use pinned memory in the data loader."""

    def __post_init__(self):
        super().__post_init__()
        self.device = torch.device(self.device)


class MOFAFLEX:
    """Fit the model using the provided data.

    Args:
        data: can be any of:

            - MuData object
            - Nested dict with group names as keys, view names as subkeys and AnnData objects as values
              (incompatible with :class:`TrainingOptions` `.group_by`)

        *args: Options for training.
    """

    def __init__(self, data: MuData | Mapping[str, Mapping[str, AnnData]], *args: Options):
        self._preprocess_options(*args)
        data = self._make_dataset(data)
        self._adjust_options(data)

        if self._data_opts.plot_data_overview:
            pl.overview(data).show()

        self._setup_likelihoods(data)
        preprocessor = self._make_preprocessor(data)

        # this needs to be after preprocessor, since preprocessor may filter out features with zero variance
        self._metadata = data.get_obs()
        self._view_names = data.view_names
        self._group_names = data.group_names
        self._sample_names = data.sample_names
        self._feature_names = data.feature_names

        self._prior_api_properties: dict[str, _PriorApiProperty] = {}

        self._fit(data, preprocessor)

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
                res, index=self.sample_names[name] if axis == 0 else self.feature_names[name], columns=factor_names
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
        for axis, priors in ((0, self._model_opts.factor_prior), (1, self._model_opts.weight_prior)):
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

    def _make_dataset(self, data: MuData | Mapping[str, Mapping[str, AnnData]]) -> MofaFlexDataset:
        return MofaFlexDataset(
            data,
            layer=self._data_opts.layer,
            group_by=self._data_opts.group_by,
            use_obs=self._data_opts.use_obs,
            use_var=self._data_opts.use_var,
            subset_var=self._data_opts.subset_var,
        )

    def _make_preprocessor(self, data: MofaFlexDataset) -> preprocessing.MofaFlexPreprocessor:
        preprocessor = preprocessing.MofaFlexPreprocessor(
            dataset=data,
            likelihoods=self._model_opts.likelihoods,
            nonnegative_weights=self._model_opts.nonnegative_weights,
            nonnegative_factors=self._model_opts.nonnegative_factors,
            scale_per_group=self._data_opts.scale_per_group,
            remove_constant_features=self._data_opts.remove_constant_features,
            state=getattr(self, "_preprocessor_state", None),
        )
        data.preprocessor = preprocessor
        return preprocessor

    def _mofaflexdataset(self, data: MuData | Mapping[str, Mapping[str, AnnData]]) -> MofaFlexDataset:
        data = self._make_dataset(data)
        self._make_preprocessor(data)
        return data

    @property
    def n_guided_factors(self) -> int:
        """Number of guided factors."""
        return self._n_guiding_vars

    @property
    def group_names(self) -> npt.NDArray[str]:
        """Group names."""
        return self._group_names

    @property
    def n_groups(self) -> int:
        """Number of groups."""
        return len(self.group_names)

    @property
    def view_names(self) -> npt.NDArray[str]:
        """View names."""
        return self._view_names

    @property
    def n_views(self) -> int:
        """Number of views."""
        return len(self.view_names)

    @property
    def feature_names(self) -> dict[str, npt.NDArray[str]]:
        """Feature names for each view."""
        return self._feature_names

    @property
    def n_features(self) -> dict[str, int]:
        """Number of features in each view."""
        return {k: len(v) for k, v in self.feature_names.items()}

    @property
    def n_features_total(self) -> int:
        """Total number of features."""
        return sum(self.n_features.values())

    @property
    def sample_names(self) -> dict[str, npt.NDArray[str]]:
        """Sample names for each group."""
        return self._sample_names

    @property
    def n_samples(self) -> dict[str, int]:
        """Number of samples in each group."""
        return {k: len(v) for k, v in self.sample_names.items()}

    @property
    def n_samples_total(self) -> int:
        """Total number of samples."""
        return sum(self.n_samples.values())

    @property
    def n_total_factors(self):
        """Total number of factors."""
        return self._model_opts.n_factors

    @property
    def n_factors(self) -> int:
        """Number of uninformed factors."""
        return self._n_factors

    @property
    def factor_order(self) -> npt.NDArray[int]:
        """Ordering of factors by explained variance (highest to lowest)."""
        return self._factor_order

    @factor_order.setter
    def factor_order(self, value: npt.ArrayLike):
        order = np.asarray(value, dtype=int)
        if order.ndim != 1:
            raise ValueError(f"The ordering must have 1 dimension, but got {order.ndim}.")
        if order.size != self.n_factors:
            raise ValueError(f"The ordering must have {self.n_factors} items, but got {order.size}.")
        if order.min() != 0 or order.max() != self.n_factors - 1 or np.unique(order).size != order.size:
            raise ValueError(f"The ordering must contain all integers in [0, {self.n_factors}).")
        self._factor_order = order

    @property
    def factor_names(self) -> npt.NDArray[str | np.str_]:
        """Factor names."""
        return self._factor_names

    @property
    def training_loss(self) -> npt.NDArray[np.float32]:
        """Total loss (negative ELBO) for each training epoch."""
        return self._train_loss_elbo

    def _setup_likelihoods(self, data):
        if (
            not isinstance(self._model_opts.likelihoods, dict | str | None)
            or isinstance(self._model_opts.likelihoods, str)
            and self._model_opts.likelihoods not in get_args(LikelihoodType)
            or isinstance(self._model_opts.likelihoods, dict)
            and not all(val in get_args(LikelihoodType) for val in self._model_opts.likelihoods.values())
        ):
            raise ValueError("Likelihoods must be a dictionary or a string containing a valid likelihood name.")

        if self._model_opts.likelihoods is None:
            self._model_opts.likelihoods = data.apply(Likelihood.infer, by_group=False)
            msg = []
            for view_name, likelihood in self._model_opts.likelihoods.items():
                msg.append(f"{view_name}: {likelihood}")
            _logger.info("No likelihoods provided. Using inferred likelihoods: " + "; ".join(msg))
        else:
            if isinstance(self._model_opts.likelihoods, str):
                self._model_opts.likelihoods = dict.fromkeys(data.view_names, self._model_opts.likelihoods)

            self._model_opts.likelihoods = {
                view: Likelihood.get(likelihood) for view, likelihood in self._model_opts.likelihoods.items()
            }

            data.apply(
                lambda *args, likelihood, **kwargs: likelihood.validate(*args, **kwargs),
                view_kwargs={"likelihood": self._model_opts.likelihoods},
                by_group=False,
            )

    def _setup_annotations(self, data):
        self._n_factors = self._model_opts.n_factors
        factor_names = [f"Factor {k + 1}" for k in range(self._model_opts.n_factors)]
        for prior in chain(self._model_opts.factor_prior, self._model_opts.weight_prior):
            factor_names = prior.adjust_factors(factor_names)

        self._model_opts.n_factors = len(factor_names)

        self._factor_names = np.asarray(factor_names)
        self._factor_order = np.arange(self._model_opts.n_factors)

    def _setup_guiding_vars(self):
        guiding_vars_names = (
            list(self._data_opts.guiding_vars_obs_keys.keys()) if self._data_opts.guiding_vars_obs_keys else []
        )
        self._n_guiding_vars = len(guiding_vars_names)

        # update global number of factors
        self._model_opts.n_factors = self._model_opts.n_factors + self._n_guiding_vars

        # update global factor names (dense factors + guiding vars + informed factors)
        self._factor_names = np.concatenate((self._factor_names, guiding_vars_names))

    def _setup_svi(
        self, init_tensor, covariates, guiding_vars_factors, guiding_vars_n_categories, feature_means, sample_means
    ):
        model = MofaFlexModel(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_factors=self._model_opts.n_factors,
            likelihoods=self._model_opts.likelihoods,
            guiding_vars_likelihoods=self._model_opts.guiding_vars_likelihoods,
            guiding_vars_n_categories=guiding_vars_n_categories,
            guiding_vars_factors=guiding_vars_factors,
            guiding_vars_scales=self._model_opts.guiding_vars_scales,
            factor_prior=self._model_opts.factor_prior,
            weight_prior=self._model_opts.weight_prior,
            nonnegative_factors=self._model_opts.nonnegative_factors,
            nonnegative_weights=self._model_opts.nonnegative_weights,
            feature_means=feature_means,
            sample_means=sample_means,
            factors_init_tensor=init_tensor,
            annotation_confidence=self._model_opts.annotation_confidence,
        ).to(self._train_opts.device)

        n_iterations = int(self._train_opts.max_epochs * (self.n_samples_total // self._train_opts.batch_size))
        gamma = 0.1
        lrd = gamma ** (1 / n_iterations)

        optimizer = ClippedAdam(model.get_lr_func(self._train_opts.lr, lrd=lrd))

        svi = SVI(
            model=pyro.poutine.scale(model.model, scale=1.0 / self.n_samples_total),
            guide=pyro.poutine.scale(model.guide, scale=1.0 / self.n_samples_total),
            optim=optimizer,
            loss=TraceMeanField_ELBO(
                retain_graph=True, num_particles=self._train_opts.n_particles, vectorize_particles=True
            ),
        )

        return svi, model

    def _post_fit(self, data, preprocessor, covariates, model, train_loss_elbo):
        self._weights = model.get_weights()
        self._factors = model.get_factors()
        self._dispersions = model.get_dispersion()
        self._train_loss_elbo = np.asarray(train_loss_elbo)

        self._preprocessor_state = preprocessor.state_dict()

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
        _logger.info(f"Initializing factors using `{self._model_opts.init_factors}` method...")

        if not isinstance(self._model_opts.init_factors, str):
            for group_name, n in self.n_samples.items():
                init_tensor[group_name]["loc"] = np.full(
                    shape=(n, self._model_opts.n_factors), fill_value=self._model_opts.init_factors, dtype=np.float32
                ).T[..., None]
                init_tensor[group_name]["scale"] = np.full(
                    shape=(n, self._model_opts.n_factors), fill_value=self._model_opts.init_scale, dtype=np.float32
                ).T[..., None]
            return init_tensor
        match self._model_opts.init_factors:
            case "random":
                for group_name, n in self.n_samples.items():
                    init_tensor[group_name]["loc"] = np.random.uniform(size=(n, self._model_opts.n_factors))
            case "orthogonal":
                for group_name, n in self.n_samples.items():
                    # Compute PCA of random vectors
                    pca = PCA(n_components=self._model_opts.n_factors, whiten=True)
                    pca.fit(stats.norm.rvs(loc=0, scale=1, size=(n, self._model_opts.n_factors)).T)
                    init_tensor[group_name]["loc"] = pca.components_.T
            case "pca" | "nmf" as init:
                if init == "pca":
                    initializer = PCA(n_components=self._model_opts.n_factors, whiten=True)
                elif init == "nmf":
                    initializer = NMF(n_components=self._model_opts.n_factors, max_iter=1000)

                inits = data.apply(
                    self._init_factor_group, by_view=False, impute_missings=impute_missings, initializer=initializer
                )
                for group_name, init in inits.items():
                    init_tensor[group_name]["loc"] = init
            case _:
                raise ValueError(
                    f"Initialization method `{self._model_opts.init_factors}` not found. Please choose from `random`, `orthogonal`, `PCA`, or `NMF`."
                )

        for group_name, n in self.n_samples.items():
            # scale factor values from -1 to 1 (per factor)
            q = init_tensor[group_name]["loc"]

            if q.shape[0] > 1:  # min and max are not defined for dimensions of size 1
                q = 2.0 * (q - np.min(q, axis=0)) / (np.max(q, axis=0) - np.min(q, axis=0)) - 1
            elif n > 0:
                q = 2.0 * (q - np.min(q)) / (np.max(q) - np.min(q)) - 1

            # Add artifical dimension at dimension -2 for broadcasting
            init_tensor[group_name]["loc"] = q.T[..., None].astype(np.float32, copy=False)
            init_tensor[group_name]["scale"] = np.full(
                shape=(n, self._model_opts.n_factors), fill_value=self._model_opts.init_scale, dtype=np.float32
            ).T[..., None]

        return init_tensor

    def _preprocess_options(self, *args: Options):
        self._data_opts = DataOptions()
        self._model_opts = ModelOptions()
        self._train_opts = TrainingOptions()
        self._gp_opts = SmoothOptions()

        for arg in args:
            match arg:
                case DataOptions():
                    self._data_opts |= arg
                case ModelOptions():
                    self._model_opts |= arg
                case TrainingOptions():
                    self._train_opts |= arg
                case SmoothOptions():
                    self._gp_opts |= arg

        if self._train_opts.seed is not None:
            try:
                self._train_opts.seed = int(self._train_opts.seed)
            except ValueError:
                _logger.warning(f"Could not convert `{self._train_opts.seed}` to integer.")
                self._train_opts.seed = None

        if self._train_opts.seed is None:
            self._train_opts.seed = int(time.strftime("%y%m%d%H%M"))

    def _adjust_options(self, data: Mapping[str, Mapping[str, AnnData]]):
        # convert input arguments to dictionaries if necessary
        if self._data_opts.guiding_vars_obs_keys is not None:
            if isinstance(self._data_opts.guiding_vars_obs_keys, str):
                self._data_opts.guiding_vars_obs_keys = [self._data_opts.guiding_vars_obs_keys]
            if isinstance(self._data_opts.guiding_vars_obs_keys, Sequence):
                self._data_opts.guiding_vars_obs_keys = {
                    obs_key: dict.fromkeys(data.group_names, obs_key)
                    for obs_key in self._data_opts.guiding_vars_obs_keys
                }
            guiding_vars_names = self._data_opts.guiding_vars_obs_keys.keys()
        else:
            guiding_vars_names = ()

        for opt_name, keys in zip(
            (
                "weight_prior",
                "factor_prior",
                "nonnegative_weights",
                "nonnegative_factors",
                "guiding_vars_likelihoods",
                "guiding_vars_scales",
            ),
            (
                data.view_names,
                data.group_names,
                data.view_names,
                data.group_names,
                guiding_vars_names,
                guiding_vars_names,
            ),
            strict=True,
        ):
            val = getattr(self._model_opts, opt_name)
            if not isinstance(val, dict):
                setattr(self._model_opts, opt_name, dict.fromkeys(keys, val))

        for opt_name, keys in zip(
            ("covariates_obs_key", "covariates_obsm_key", "annotations_varm_key"),
            (data.group_names, data.group_names, data.view_names),
            strict=True,
        ):
            val = getattr(self._data_opts, opt_name)
            if isinstance(val, str):
                setattr(self._data_opts, opt_name, dict.fromkeys(keys, val))

        self._train_opts.device = self._setup_device(self._train_opts.device)
        if self._train_opts.batch_size is None or not (0 < self._train_opts.batch_size <= data.n_samples_total):
            self._train_opts.batch_size = data.n_samples_total

        factor_prior_groups = defaultdict(list)
        for group_name, prior in self._model_opts.factor_prior.items():
            factor_prior_groups[prior].append(group_name)
        self._model_opts.factor_prior = []
        for priorname, gnames in factor_prior_groups.items():
            prior = Prior(
                priorname,
                axis=0,
                names=gnames,
                covariates_obs_key=self._data_opts.covariates_obs_key,
                covariates_obsm_key=self._data_opts.covariates_obsm_key,
                options=self._gp_opts,
            )
            self._model_opts.factor_prior.append(prior)

        weight_prior_groups = defaultdict(list)
        for view_name, prior in self._model_opts.weight_prior.items():
            weight_prior_groups[prior].append(view_name)
        self._model_opts.weight_prior = [
            Prior(prior, axis=1, names=gnames, annotations_varm_key=self._data_opts.annotations_varm_key)
            for prior, gnames in weight_prior_groups.items()
        ]

    def _fit(self, data, preprocessor):
        pyro.set_rng_seed(self._train_opts.seed)

        datasets = {"data": data}
        for prior in chain(self._model_opts.factor_prior, self._model_opts.weight_prior):
            if priordsets := prior.get_datasets(data):
                datasets.update(priordsets)

        # this needs to run after prior.get_datasets()
        self._setup_annotations(data)
        self._setup_guiding_vars()

        guiding_vars_factors = {
            self.factor_names[self._model_opts.n_factors - self.n_guided_factors + i]: self._model_opts.n_factors
            - self.n_guided_factors
            + i
            for i in range(self.n_guided_factors)
        }

        # get unique categories for each guiding variable
        guiding_vars_n_categories = {}
        if self.n_guided_factors > 0:
            datasets["guiding_vars"] = guiding_vars = GuidingVarsDataset(data, self._data_opts.guiding_vars_obs_keys)

            for guiding_var_name, guiding_var_likelihood in self._model_opts.guiding_vars_likelihoods.items():
                if guiding_var_likelihood == "Categorical":
                    guiding_vars_categories = set()
                    # find number of unique categories across groups
                    for group_name in self._group_names:
                        guiding_vars_categories.update(
                            guiding_vars.datasets[guiding_var_name].covariates[group_name].iloc[:, 0].to_list()
                        )
                    guiding_vars_n_categories[guiding_var_name] = len(guiding_vars_categories)

                else:
                    # if not categorical, set to default
                    guiding_vars_n_categories[guiding_var_name] = 0

        init_tensor = self._initialize_factors(data)

        covariates = datasets.get("gp_covariates")
        svi, model = self._setup_svi(
            init_tensor,
            covariates.covariates if covariates else None,
            guiding_vars_factors,
            guiding_vars_n_categories,
            preprocessor.feature_means,
            preprocessor.sample_means,
        )

        # clean start
        pyro.enable_validation(True)
        pyro.clear_param_store()

        # Train
        singlebatch = self._train_opts.batch_size >= max(self.n_samples.values())
        collate_fn_map = {
            torch.Tensor: lambda x, **kwargs: x[0].to(self._train_opts.device, non_blocking=True),
            slice: lambda x, **kwargs: x[0],
        }
        dataset = StackDataset(**datasets)
        if singlebatch:
            batch = collate(
                (default_convert(dataset.__getitems__(sample_all_data_as_one_batch(data))),),
                collate_fn_map=collate_fn_map,
            )
            batchdata = batch.pop("data")
        else:
            loader = DataLoader(
                dataset,
                batch_sampler=MofaFlexBatchSampler(
                    data.n_samples, self._train_opts.batch_size, False, generator=torch.default_generator
                ),
                collate_fn=default_convert,
                num_workers=self._train_opts.num_workers,
                pin_memory=self._train_opts.pin_memory,
                persistent_workers=self._train_opts.num_workers > 0,
            )

        train_loss_elbo = []
        earlystopper = EarlyStopper(
            mode="min", min_delta=0.1, patience=self._train_opts.early_stopper_patience, percentage=True
        )
        with self._train_opts.device:
            for prior in chain(self._model_opts.factor_prior, self._model_opts.weight_prior):
                prior.on_train_start()

        with tqdm(range(self._train_opts.max_epochs), unit="epochs", dynamic_ncols=True) as t:
            for i in t:
                with self._train_opts.device, torch.inference_mode():
                    for prior in chain(self._model_opts.factor_prior, self._model_opts.weight_prior):
                        prior.on_train_epoch_start(i)

                epoch_loss = 0
                if singlebatch:
                    with self._train_opts.device:
                        epoch_loss += svi.step(**batchdata, **batch)
                else:
                    for batch in loader:
                        batch = collate((batch,), collate_fn_map=collate_fn_map)
                        with self._train_opts.device:
                            epoch_loss += svi.step(**batch.pop("data"), **batch)

                with self._train_opts.device, torch.inference_mode():
                    for prior in chain(self._model_opts.factor_prior, self._model_opts.weight_prior):
                        prior.on_train_epoch_end(i)

                train_loss_elbo.append(epoch_loss)
                t.set_postfix({"Loss": epoch_loss}, refresh=False)

                if earlystopper.step(epoch_loss):
                    _logger.info(f"Training converged after {i} epochs.")
                    break

            if isinstance(t, tqdm_notebook):  # https://github.com/tqdm/tqdm/issues/1659
                t.container.children[1].bar_style = "success"

            self._post_fit(data, preprocessor, covariates, model, train_loss_elbo)

            with self._train_opts.device, torch.inference_mode():
                for prior in chain(self._model_opts.factor_prior, self._model_opts.weight_prior):
                    if prior.axis == 0:
                        kwargs = {
                            "results": self._factors,
                            "results_nonnegative": self._model_opts.nonnegative_factors,
                            "nonfactor_names": self.sample_names,
                        }
                    else:
                        kwargs = {
                            "results": self._weights,
                            "results_nonnegative": self._model_opts.nonnegative_weights,
                            "nonfactor_names": self.feature_names,
                        }
                    prior.on_train_end(
                        data, factor_names=self.factor_names, batch_size=self._train_opts.batch_size, **kwargs
                    )

        self._df_r2_full, self._df_r2_factors, self._factor_order = self._sort_factors(
            data,
            factors=self._get_postprocessed_factors(moment="mean", sparse_type="mix", ordered=False),
            weights=self._get_postprocessed_weights(moment="mean", sparse_type="mix", ordered=False),
        )

        if self._train_opts.save_path is not False:
            if self._train_opts.save_path is None:
                self._train_opts.save_path = f"mofaflex_{time.strftime('%Y%m%d_%H%M%S')}.h5"
            else:
                self._train_opts.save_path = str(self._train_opts.save_path)
            _logger.info(f"Saving results to {self._train_opts.save_path}...")
            Path(self._train_opts.save_path).parent.mkdir(parents=True, exist_ok=True)
            self._save(self._train_opts.save_path, self._train_opts.mofa_compat, data, preprocessor.feature_means)

        self._init_api()

    def _sort_factors(self, data, factors, weights, subsample=1000):
        # Loop over all groups
        dfs_factors, dfs_full = {}, {}

        def r2_wrapper(view, group_name, view_name):
            if subsample is not None and subsample > 0 and subsample < view.n_obs:
                sample_idx = np.random.choice(view.n_obs, subsample, replace=False)
            else:
                sample_idx = slice(None)
            cdata = data.preprocessor(view.X[sample_idx, :], slice(None), slice(None), group_name, view_name)[0]
            if issparse(cdata):
                cdata = cdata.toarray()

            dispersions = self._dispersions.mean.get(view_name)
            if dispersions is not None:
                dispersions = align_global_array_to_local(dispersions, group_name, view_name, align_to="features")  # noqa F821
            try:
                return self._model_opts.likelihoods[view_name].r2(
                    view_name,
                    y_true=cdata,
                    factors=align_global_array_to_local(  # noqa F821
                        factors[group_name], group_name, view_name, align_to="samples", axis=0
                    )[sample_idx, :],
                    weights=align_global_array_to_local(  # noqa F821
                        weights[view_name], group_name, view_name, align_to="features", axis=0
                    ),
                    dispersions=dispersions,
                    sample_means=align_global_array_to_local(  # noqa F821
                        data.preprocessor.sample_means[group_name][view_name],
                        group_name,
                        view_name,
                        align_to="samples",
                        axis=0,
                    )[sample_idx],
                )
            except NotImplementedError:
                _logger.warning(
                    f"R2 calculation for {self._model_opts.likelihoods[view_name]} likelihood has not yet been implemented. Skipping view {view_name} for group {group_name}."
                )

        r2s = data.apply(r2_wrapper)
        for group_name, group_r2 in r2s.items():
            group_r2_factors, group_r2_full = {}, {}
            for view_name, view_r2 in group_r2.items():
                group_r2_full[view_name], group_r2_factors[view_name] = view_r2
            if len(group_r2_factors) == 0:
                _logger.warning(f"No R2 values found for group {group_name}. Skipping...")
                continue
            dfs_factors[group_name] = pd.DataFrame(group_r2_factors)
            dfs_full[group_name] = pd.Series(group_r2_full)

        # sum the R2 values across all groups
        df_concat = pd.concat(dfs_factors.values())
        df_sum = df_concat.groupby(df_concat.index).sum()
        dfs_full = pd.DataFrame(dfs_full)

        try:
            # sort factors according to mean R2 across all views
            sorted_r2_means = df_sum.mean(axis=1).sort_values(ascending=False)
            factor_order = sorted_r2_means.index.to_numpy()
        except NameError:
            _logger.warning("Sorting factors failed. Using default order.")
            factor_order = np.array(list(range(self.model_opts.n_factors)))

        return dfs_full, dfs_factors, factor_order

    def _get_postprocessed_factors(self, moment: Literal["mean", "std"] = "mean", **kwargs) -> dict[str, np.ndarray]:
        factors = {}
        for prior in self._model_opts.factor_prior:
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

    def get_r2(self, total: bool = False, ordered: bool = False) -> pd.DataFrame | dict[str, pd.DataFrame]:
        """Get the fraction of explained variance for each view and group.

        Args:
            total: If `True`, returns a DataFrame with fraction of explained variance for the full
                model for each group (columns) and view (rows). Otherwise returns a dict with group
                names as keys containing DataFrames with the fraction of explained variance for each
                view (columns) and factor(rows).
            ordered: Whether to return the factors ordered by explained variance (highest to lowest).
                Has no effect if `total == True`.
        """
        if total:
            return self._df_r2_full
        else:
            return {
                group_name: df.set_index(self.factor_names).iloc[self.factor_order if ordered else slice(None), :]
                for group_name, df in self._df_r2_factors.items()
            }

    def _get_postprocessed_weights(self, moment: Literal["mean", "std"] = "mean", **kwargs) -> dict[str, np.ndarray]:
        weights = {}
        for prior in self._model_opts.weight_prior:
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

    def get_dispersion(self, moment: Literal["mean", "std"] = "mean") -> dict[str, pd.Series]:
        """Get the dispersion vectors for each view.

        Args:
            moment: Which moment of the posterior distribution to return.
        """
        return {
            view_name: pd.Series(view_dispersion, index=self.feature_names[view_name])
            for view_name, view_dispersion in getattr(self._dispersions, moment).items()
        }

    def _setup_device(self, device):
        device = torch.device(device)
        tens = torch.tensor(())
        try:
            tens.to(device)
        except (RuntimeError, AssertionError):
            default_device = tens.device
            _logger.warning(f"Device {str(device)} is not available. Using default device: {default_device}")
            device = default_device

        return device

    def impute_data(
        self, data: MuData | Mapping[str, Mapping[str, AnnData]], missing_only=False
    ) -> dict[dict[str, AnnData]]:
        """Impute values in the training data using the trained factorization.

        Args:
            data: The data the model was trained on.
            missing_only: Only impute missing values in the data.

        Returns:
            Nested dictionary of AnnData objects with either fully imputed data or with only the missing values filled in.
            In both cases, the returned data will be preprocessed. In the case of Gaussian distributed data, that involves
            centering and scaling.
        """
        data = self._mofaflexdataset(data)

        return data.apply(
            impute,
            view_kwargs={
                "weights": self._weights.mean,
                "feature_names": self.feature_names,
                "likelihood": self._model_opts.likelihoods,
            },
            group_kwargs={"factors": self._factors.mean, "sample_names": self.sample_names},
            missingonly=missing_only,
            preprocessor=data.preprocessor,
        )

    def _save(
        self,
        path: str | Path,
        mofa_compat: MOFACompatOption = False,
        data: Mapping[str, Mapping[str, AnnData]] | None = None,
        intercepts: Mapping[str, Mapping[str, np.ndarray]] | None = None,
    ):
        state = {
            "weights": self._weights._asdict(),
            "factors": self._factors._asdict(),
            "n_guiding_vars": self._n_guiding_vars,
            "df_r2_full": self._df_r2_full,
            "df_r2_factors": self._df_r2_factors,
            "n_factors": self._n_factors,
            "factor_names": self._factor_names,
            "factor_order": self._factor_order,
            "dispersions": self._dispersions._asdict(),
            "train_loss_elbo": self._train_loss_elbo,
            "group_names": self._group_names,
            "view_names": self._view_names,
            "feature_names": self._feature_names,
            "sample_names": self._sample_names,
            "metadata": self._metadata,
            "data_opts": self._data_opts.asdict(),
            "model_opts": self._model_opts.asdict(),
            "train_opts": self._train_opts.asdict(),
            "gp_opts": self._gp_opts.asdict(),
            "preprocessor_state": self._preprocessor_state,
        }
        state["train_opts"]["device"] = str(state["train_opts"]["device"])
        state["model_opts"]["likelihoods"] = {
            view_name: str(likelihood) for view_name, likelihood in state["model_opts"]["likelihoods"].items()
        }
        state["model_opts"]["factor_prior"] = {
            str(i): prior.save() for i, prior in enumerate(state["model_opts"]["factor_prior"])
        }
        state["model_opts"]["weight_prior"] = {
            str(i): prior.save() for i, prior in enumerate(state["model_opts"]["weight_prior"])
        }

        save_model(state, path, mofa_compat, self, data, intercepts)

    @classmethod
    def load(cls, path: str | Path, map_location=None) -> "MOFAFLEX":
        """Load a saved MOFAFLEX model.

        Args:
            path: Path to the saved model file.
            map_location: Specify how to remap storage locations for PyTorch tensors. See the `torch.load`
                documentation for details.
        """
        state = load_model(path)

        if map_location is not None:
            state["train_opts"]["device"] = map_location
        state["model_opts"]["likelihoods"] = {
            view_name: Likelihood.get(likelihood)
            for view_name, likelihood in state["model_opts"]["likelihoods"].items()
        }

        model = cls.__new__(cls)
        model._weights = MeanStd(**state["weights"])
        model._factors = MeanStd(**state["factors"])
        model._n_guiding_vars = state.get("n_guiding_vars")
        model._df_r2_full = state["df_r2_full"]
        model._df_r2_factors = state["df_r2_factors"]
        model._n_factors = state["n_factors"]
        model._factor_names = state["factor_names"]
        model._factor_order = state["factor_order"]
        model._dispersions = MeanStd(**state["dispersions"])
        model._train_loss_elbo = state["train_loss_elbo"]
        model._group_names = state["group_names"]
        model._view_names = state["view_names"]
        model._feature_names = state["feature_names"]
        model._sample_names = state["sample_names"]
        model._annotations = state.get("annotations")
        model._metadata = state["metadata"]
        model._data_opts = DataOptions(**state["data_opts"])
        model._model_opts = ModelOptions(**state["model_opts"])
        model._train_opts = TrainingOptions(**state["train_opts"])
        model._gp_opts = SmoothOptions(**state["gp_opts"])
        model._preprocessor_state = state["preprocessor_state"]

        model._model_opts.factor_prior = [
            Prior.load(state, model.n_total_factors, model.n_samples, map_location=map_location)
            for state in model._model_opts.factor_prior.values()
        ]
        model._model_opts.weight_prior = [
            Prior.load(state, model.n_total_factors, model.n_features, map_location=map_location)
            for state in model._model_opts.weight_prior.values()
        ]

        model._prior_api_properties = {}
        model._init_api()

        return model


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

    getters = MOFAFLEX.get_factors, MOFAFLEX.get_weights
    getter_sigs = tuple(inspect.signature(getter) for getter in getters)
    getter_params = tuple([param for param in sig.parameters.values() if param.name != "kwargs"] for sig in getter_sigs)
    getter_annots = tuple(getter.__annotations__ for getter in getters)
    getter_docs = [getter.__doc__ for getter in getters]
    getter_indents = [" " * get_indentation(doc) for doc in getter_docs]

    for axis, axisname, priors in (
        (0, "factor", Prior.known_factor_priors()),
        (1, "weight", Prior.known_weight_priors()),
    ):
        namescount = Counter()
        for api in chain(*(Prior.class_(x).api() for x in priors)):
            namescount[api.name] += 1
        duplicates = {k for k, v in namescount.items() if v > 1}

        for prior in priors:
            priorcls = Prior.class_(prior)
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
                    setattr(MOFAFLEX, name, attr)
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
                    wrapperfunc.__qualname__ = f"{MOFAFLEX.__qualname__}.{name}"
                    wrapperfunc.__name__ = name
                setattr(MOFAFLEX, name, wrapperfunc)

                postprocess_method = priorcls.postprocess_results
                params = [
                    param
                    for param in inspect.signature(postprocess_method).parameters.values()
                    if param.name not in {"self", "results", "moment", "kwargs"}
                ]
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
        setattr(MOFAFLEX, method.__name__, wrapper)

    return apinames


_apinames = _init_api()
