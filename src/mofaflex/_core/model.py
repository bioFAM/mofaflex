from collections import defaultdict
from operator import attrgetter

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.infer.autoguide.guides import deep_setattr
from pyro.nn import PyroModule, PyroModuleList, PyroParam

from .gp import GP
from .likelihoods import Likelihood
from .pyro.likelihoods import PyroLikelihood
from .pyro.priors import Prior
from .utils import FactorPrior, MeanStd, WeightPrior

EPS = 1e-8

PyroModuleDict = PyroModule[torch.nn.ModuleDict]


class Generative(PyroModule):
    def __init__(
        self,
        n_samples: dict[str, int],
        n_features: dict[str, int],
        n_factors: int,
        likelihoods: dict[str, Likelihood],
        guiding_vars_likelihoods: dict[str, str] | None = None,
        guiding_vars_n_categories: dict[str, int] | None = None,
        guiding_vars_factors: dict[str, int] | None = None,
        guiding_vars_scales: dict[str, float] | None = None,
        prior_scales=None,
        factor_prior: dict[str, FactorPrior] | FactorPrior = "Normal",
        weight_prior: dict[str, WeightPrior] | WeightPrior = "Normal",
        nonnegative_weights: dict[str, bool] | bool = False,
        nonnegative_factors: dict[str, bool] | bool = False,
        feature_means: dict[dict[str, torch.Tensor]] = None,
        sample_means: dict[dict[str, torch.Tensor]] = None,
        gp: GP | None = None,
        gp_group_names: list[str] | None = None,
        **kwargs,
    ):
        super().__init__("Generative")

        self.group_names = tuple(n_samples.keys())
        self.view_names = tuple(n_features.keys())

        if isinstance(factor_prior, str):
            factor_prior = dict.fromkeys(self.group_names, factor_prior)

        if isinstance(weight_prior, str):
            weight_prior = dict.fromkeys(self.view_names, weight_prior)

        if isinstance(nonnegative_weights, bool):
            nonnegative_weights = dict.fromkeys(self.view_names, nonnegative_weights)

        if isinstance(nonnegative_factors, bool):
            nonnegative_factors = dict.fromkeys(self.group_names, nonnegative_factors)

        self.n_samples = n_samples
        self.n_features = n_features
        self.n_factors = n_factors
        self.factor_prior = factor_prior
        self.weight_prior = weight_prior
        self.likelihoods = PyroModuleDict(
            {
                view_name: likelihood.pyro_likelihood(
                    view_name=view_name,
                    sample_dim=self._sample_plate_dim,
                    feature_dim=self._feature_plate_dim,
                    sample_means=sample_means,
                    feature_means=feature_means,
                    is_guiding_var=False,
                )
                for view_name, likelihood in likelihoods.items()
            }
        )
        self.guiding_vars_names = guiding_vars_factors.keys()
        self.guiding_vars_likelihoods = PyroModuleDict(
            {
                guiding_var_name: PyroLikelihood(
                    guiding_vars_likelihoods[guiding_var_name],
                    view_name=guiding_var_name,
                    sample_dim=self._sample_plate_dim,
                    feature_dim=self._feature_plate_dim,
                    sample_means=sample_means,
                    feature_means={"dummy_name": {guiding_var_name: torch.zeros(1, 1)}},
                )
                for guiding_var_name in self.guiding_vars_names
            }
        )
        self.guiding_vars_n_categories = guiding_vars_n_categories
        self.guiding_vars_factors = guiding_vars_factors
        self.nonnegative_weights = nonnegative_weights
        self.nonnegative_factors = nonnegative_factors
        self.guiding_vars_scales = guiding_vars_scales

        factor_prior_groups = defaultdict(list)
        for group_name, prior in factor_prior.items():
            factor_prior_groups[prior].append(group_name)
        self.factors = PyroModuleList(
            [
                Prior(
                    prior,
                    names=groups,
                    factor_dim=-3,
                    nonfactor_dim=self._sample_plate_dim,
                    n_factors=n_factors,
                    n_nonfactors=n_samples,
                    gp=gp,
                )
                for prior, groups in factor_prior_groups.items()
            ]
        )

        weight_prior_groups = defaultdict(list)
        for view_name, prior in weight_prior.items():
            weight_prior_groups[prior].append(view_name)
        self.weights = PyroModuleList(
            [
                Prior(
                    prior,
                    names=views,
                    factor_dim=-3,
                    nonfactor_dim=self._feature_plate_dim,
                    n_factors=n_factors,
                    n_nonfactors=n_features,
                    prior_scales=prior_scales,
                )
                for prior, views in weight_prior_groups.items()
            ]
        )

        self.pos_transform = torch.nn.ReLU()

        self.scale_elbo = True
        n_views = len(self.view_names)
        self.view_scales = dict.fromkeys(self.view_names, 1.0)
        if self.scale_elbo and n_views > 1:
            for view_name, view_n_features in n_features.items():
                self.view_scales[view_name] = (n_views / (n_views - 1)) * (
                    1.0 - view_n_features / sum(n_features.values())
                )

        self._setup_distributions()

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _get_prior_scale(self, view_name: str):
        return getattr(self, f"prior_scales_{view_name}", None)

    _sample_plate_dim = -1
    _feature_plate_dim = -2

    def _get_plates(self, subsample=None):
        sample_plates = {}

        for group_name in self.group_names:
            sample_plates[group_name] = pyro.plate(
                f"plate_samples_{group_name}",
                self.n_samples[group_name],
                dim=self._sample_plate_dim,
                subsample=subsample[group_name],
            )

        feature_plates = {}
        for view_name in self.view_names:
            feature_plates[view_name] = pyro.plate(
                f"plate_features_{view_name}",
                self.n_features[view_name],
                subsample=torch.arange(  # workaround for https://github.com/pyro-ppl/pyro/pull/3405
                    self.n_features[view_name]
                ),
                dim=self._feature_plate_dim,
            )

        guiding_var_plate = pyro.plate("plate_guiding_vars", 1, subsample=torch.arange(1), dim=self._feature_plate_dim)

        factors_plate = pyro.plate("plate_factors", self.n_factors, dim=-3)

        return sample_plates, feature_plates, guiding_var_plate, factors_plate

    def _setup_distributions(self):
        self.sample_guiding_vars_weights = {}
        for guiding_var_name in self.guiding_vars_names:
            self.sample_guiding_vars_weights[guiding_var_name] = self._sample_guiding_vars_weights_normal

    def _sample_guiding_vars_weights_normal(self, guiding_var_name, **kwargs):
        weights_dim = self.guiding_vars_n_categories[guiding_var_name]
        return pyro.sample(
            f"guiding_vars_w_{guiding_var_name}",
            dist.Normal(torch.zeros(weights_dim, 2), torch.ones(weights_dim, 2)).to_event(
                2
            ),  # (categories, intercept & slope)
        )

    def forward(self, data, sample_idx, nonmissing_samples, nonmissing_features, covariates, guiding_vars):
        sample_plates, feature_plates, guiding_var_plate, factor_plate = self._get_plates(subsample=sample_idx)

        factors = {}
        for prior in self.factors:
            factors.update(prior.model(factor_plate, sample_plates, covariates=covariates))

        for group_name, group_factors in factors.items():
            if self.nonnegative_factors[group_name]:
                factors[group_name] = self.pos_transform(group_factors)

        weights = {}
        for prior in self.weights:
            weights.update(prior.model(factor_plate, feature_plates))

        for view_name, view_weights in weights.items():
            if self.nonnegative_weights[view_name]:
                weights[view_name] = self.pos_transform(view_weights)

        # sample guiding variable weights
        for guiding_var_name in self.guiding_vars_names:
            self.sample_dict[f"w_guiding_vars_{guiding_var_name}"] = self.sample_guiding_vars_weights[guiding_var_name](
                guiding_var_name
            )

        # sample observations
        for group_name, group in data.items():
            gnonmissing_samples = nonmissing_samples[group_name]
            gnonmissing_features = nonmissing_features[group_name]
            for view_name, view_obs in group.items():
                if view_obs.numel() == 0:  # can occur in the last batch of an epoch if the batch is small
                    continue

                vnonmissing_samples = gnonmissing_samples[view_name]
                vnonmissing_features = gnonmissing_features[view_name]

                z = factors[group_name][..., vnonmissing_samples]
                w = weights[view_name][..., vnonmissing_features, :]

                loc = torch.einsum("...ijk,...ilj->...jlk", z, w)

                obs = view_obs.T
                self.likelihoods[view_name].model(
                    data=obs,
                    estimate=loc,
                    group_name=group_name,
                    scale=self.view_scales[view_name],
                    sample_plate=sample_plates[group_name],
                    feature_plate=feature_plates[view_name],
                    nonmissing_samples=vnonmissing_samples,
                    nonmissing_features=vnonmissing_features,
                )

            # guiding variables
            for guiding_var_name in self.guiding_vars_names:
                if group_name not in guiding_vars[guiding_var_name]:
                    continue

                z_guiding = factors[group_name][self.guiding_vars_factors[guiding_var_name], 0]
                w_guiding = self.sample_dict[f"w_guiding_vars_{guiding_var_name}"]

                # (n_cats, 1) + (n_cats, 1) * (n_samples,)
                loc = w_guiding[:, 0, None] + w_guiding[:, 1, None] * z_guiding  # (n_cats, n_samples)
                obs_guiding_vars = guiding_vars[guiding_var_name][group_name].squeeze(-1)

                self.guiding_vars_likelihoods[guiding_var_name].model(
                    data=obs_guiding_vars,
                    estimate=loc,
                    group_name=group_name,
                    scale=self.guiding_vars_scales[guiding_var_name],
                    sample_plate=sample_plates[group_name],
                    feature_plate=guiding_var_plate,
                    nonmissing_samples=slice(None),
                    nonmissing_features=slice(None),
                )

        return self.sample_dict


class Variational(PyroModule):
    def __init__(
        self,
        generative,
        z_init_tensor: dict = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        init_prob: float = 0.5,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        init_shape: float = 10,
        init_rate: float = 10,
        **kwargs,
    ):
        super().__init__("Variational")
        self.generative = generative
        self.locs = PyroModule()
        self.scales = PyroModule()
        self.probs = PyroModule()
        self.alphas = PyroModule()
        self.betas = PyroModule()
        self.shapes = PyroModule()
        self.rates = PyroModule()

        self.init_loc = init_loc
        self.init_scale = init_scale
        self.init_prob = init_prob
        self.init_alpha = init_alpha
        self.init_beta = init_beta
        self.init_shape = init_shape
        self.init_rate = init_rate
        self.z_init_tensor = z_init_tensor

        self._setup_parameters()
        self._setup_distributions()

        self.sample_dict: dict[str, torch.Tensor] = {}

    def _get_loc_and_scale(self, site_name):
        site_loc = attrgetter(site_name)(self.locs)
        site_scale = attrgetter(site_name)(self.scales)
        return MeanStd(site_loc, site_scale)

    def _get_gp_loc_and_scale(self, group: str | None = None):
        if not len(self.generative.gp_group_names):
            return {}, {}

        loc = attrgetter("z_gp")(self.locs)
        scale = attrgetter("z_gp")(self.scales)

        gp_group_sizes = [self.generative.n_samples[g] for g in self.generative.gp_group_names]
        if group is not None:
            gp_group_offsets = torch.as_tensor([0] + gp_group_sizes).cumsum()
            group_idx = self.generative.get_gp_group_idx(group)
            offset = slice(gp_group_offsets[group_idx], gp_group_offsets[group_idx + 1])
            site_loc = offset
            site_scale = scale[..., offset]
        else:
            site_loc = dict(zip(self.generative.gp_group_names, torch.split(loc, gp_group_sizes, dim=-1), strict=False))
            site_scale = dict(
                zip(self.generative.gp_group_names, torch.split(scale, gp_group_sizes, dim=-1), strict=False)
            )

        return MeanStd(site_loc, site_scale)

    def _get_prob(self, site_name: str):
        site_prob = attrgetter(site_name)(self.probs)
        return site_prob

    def _get_alpha_and_beta(self, site_name: str):
        site_alpha = attrgetter(site_name)(self.alphas)
        site_beta = attrgetter(site_name)(self.betas)
        return site_alpha, site_beta

    def _get_shape_and_rate(self, site_name: str):
        site_shape = attrgetter(site_name)(self.shapes)
        site_rate = attrgetter(site_name)(self.rates)
        return site_shape, site_rate

    def _setup_parameters(self):
        # guiding variables variational parameters
        for guiding_var_name in self.generative.guiding_vars_names:
            deep_setattr(
                self.locs,
                f"guiding_vars_w_{guiding_var_name}",
                PyroParam(
                    torch.full([self.generative.guiding_vars_n_categories[guiding_var_name], 2], self.init_loc),
                    constraint=constraints.real,
                ),
            )
            deep_setattr(
                self.scales,
                f"guiding_vars_w_{guiding_var_name}",
                PyroParam(
                    torch.full([self.generative.guiding_vars_n_categories[guiding_var_name], 2], self.init_scale),
                    constraint=constraints.softplus_positive,
                ),
            )

    def _setup_distributions(self):
        # guiding variables
        self.sample_guiding_vars_weights = {}
        for guiding_var_name in self.generative.guiding_vars_names:
            self.sample_guiding_vars_weights[guiding_var_name] = self._sample_guiding_vars_weights_normal

    def _sample_guiding_vars_weights_normal(self, guiding_var_name, **kwargs):
        w_loc, w_scale = self._get_loc_and_scale(f"guiding_vars_w_{guiding_var_name}")
        return pyro.sample(f"guiding_vars_w_{guiding_var_name}", dist.Normal(w_loc, w_scale).to_event(2))

    def forward(self, data, sample_idx, nonmissing_samples, nonmissing_features, covariates, guiding_vars):
        sample_plates, feature_plates, guiding_var_plate, factor_plate = self.generative._get_plates(
            subsample=sample_idx
        )

        for prior in self.generative.factors:
            prior.guide(factor_plate, sample_plates, covariates=covariates)

        for prior in self.generative.weights:
            prior.guide(factor_plate, feature_plates)

        for guiding_var_name in self.generative.guiding_vars_names:
            self.sample_dict[f"guiding_vars_w_{guiding_var_name}"] = self.sample_guiding_vars_weights[guiding_var_name](
                guiding_var_name
            )

        for group_name, group in data.items():
            for view_name in group.keys():
                self.generative.likelihoods[view_name].guide(
                    group_name, sample_plates[group_name], feature_plates[view_name]
                )

            for guiding_var_name in self.generative.guiding_vars_names:
                if group_name in guiding_vars[guiding_var_name]:
                    self.generative.guiding_vars_likelihoods[guiding_var_name].guide(
                        group_name, sample_plates[group_name], guiding_var_plate
                    )

        return self.sample_dict

    def get_lr_func(self, base_lr: float, **kwargs):
        modifiers = {}
        for i, prior in enumerate(self.generative.weights):
            modifiers.update(
                {
                    f"{__class__.__name__}.generative.weights.{i}.{pname}": mod
                    for pname, mod in prior.learning_rate_multipliers
                }
            )
        for i, prior in enumerate(self.generative.factors):
            modifiers.update(
                {
                    f"{__class__.__name__}.generative.factors.{i}.{pname}": mod
                    for pname, mod in prior.learning_rate_multipliers
                }
            )

        def lr_func(param_name):
            return dict(lr=base_lr * modifiers.get(param_name, 1), **kwargs)

        return lr_func

    @torch.inference_mode()
    def get_factors(self):
        """Get all factor matrices, z_x."""
        factors = MeanStd({}, {})
        for prior in self.generative.factors:
            for lsidx, vals in enumerate(prior.posterior):
                factors[lsidx].update(vals)

        for group_name in self.generative.group_names:
            if self.generative.nonnegative_factors[group_name]:
                factors.mean[group_name] = self.generative.pos_transform(factors.mean[group_name])
            factors.mean[group_name] = factors.mean[group_name].cpu().numpy()
            factors.std[group_name] = factors.std[group_name].cpu().numpy()

        return factors

    @torch.inference_mode()
    def get_sparse_factor_precisions(self):
        alphas = MeanStd({}, {})
        for prior in self.generative.factors:
            try:
                precisions = prior.posterior_precision
            except AttributeError:
                continue
            for group_name in precisions.shape.keys():
                d = dist.Gamma(shape=precisions.shape[group_name], rate=precisions.rate[group_name])
                alphas.mean[group_name] = d.mean.cpu().numpy()
                alphas.std[group_name] = d.stddev.cpu().numpy()
        return alphas

    @torch.inference_mode()
    def get_sparse_factor_probabilities(self):
        probs = {}
        for prior in self.generative.factors:
            try:
                for group_name, prob in prior.posterior_probability.items():
                    probs[group_name] = prob.cpu().numpy()
            except AttributeError:
                continue
        return probs

    @torch.inference_mode()
    def get_weights(self):
        """Get all weight matrices, w_x."""
        weights = MeanStd({}, {})
        for prior in self.generative.weights:
            for lsidx, vals in enumerate(prior.posterior):
                weights[lsidx].update(vals)

        for view_name in self.generative.view_names:
            if self.generative.nonnegative_weights[view_name]:
                weights.mean[view_name] = self.generative.pos_transform(weights.mean[view_name])
            weights.mean[view_name] = weights.mean[view_name].cpu().numpy()
            weights.std[view_name] = weights.std[view_name].cpu().numpy()

        return weights

    @torch.inference_mode()
    def get_sparse_weight_precisions(self):
        alphas = MeanStd({}, {})
        for prior in self.generative.weights:
            try:
                precisions = prior.posterior_precision
            except AttributeError:
                continue
            for view_name in precisions.shape.keys():
                d = dist.Gamma(shape=precisions.shape[view_name], rate=precisions.rate[view_name])
                alphas.mean[view_name] = d.mean.cpu().numpy()
                alphas.std[view_name] = d.stddev.cpu().numpy()
        return alphas

    @torch.inference_mode()
    def get_sparse_weight_probabilities(self):
        probs = {}
        for prior in self.generative.weights:
            try:
                for view_name, prob in prior.posterior_probability.items():
                    probs[view_name] = prob.cpu().numpy()
            except AttributeError:
                continue
        return probs

    @torch.inference_mode()
    def get_dispersion(self):
        """Get all dispersion vectors, dispersion_x."""
        dispersion = MeanStd({}, {})
        for view_name, likelihood in self.generative.likelihoods.items():
            try:
                disp = likelihood.dispersion
            except AttributeError:
                continue
            dispersion.mean[view_name] = disp.mean
            dispersion.std[view_name] = disp.std

        return dispersion
