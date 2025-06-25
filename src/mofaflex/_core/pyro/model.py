from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import pyro
import pyro.distributions as dist
import torch
from pyro.distributions import constraints
from pyro.nn import PyroModule, PyroModuleList, PyroParam, pyro_method

from ..utils import MeanStd
from .likelihoods import PyroLikelihood
from .priors import Prior
from .utils import PyroModuleDict, PyroParameterDict

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Literal

    from numpy.typing import NDArray

    from ..gp import GP
    from ..likelihoods import Likelihood
    from .priors import FactorPriorType, WeightPriorType


class MofaFlexModel(PyroModule):
    def __init__(
        self,
        n_samples: Mapping[str, int],
        n_features: Mapping[str, int],
        n_factors: int,
        likelihoods: Mapping[str, Likelihood],
        guiding_vars_likelihoods: Mapping[str, str] | None = None,
        guiding_vars_n_categories: Mapping[str, int] | None = None,
        guiding_vars_factors: Mapping[str, int] | None = None,
        guiding_vars_scales: Mapping[str, float] | None = None,
        prior_scales=None,
        factor_prior: Mapping[str, FactorPriorType] | FactorPriorType = "Normal",
        weight_prior: Mapping[str, WeightPriorType] | WeightPriorType = "Normal",
        nonnegative_weights: Mapping[str, bool] | bool = False,
        nonnegative_factors: Mapping[str, bool] | bool = False,
        feature_means: Mapping[str, Mapping[str, NDArray]] = None,
        sample_means: Mapping[str, Mapping[str, NDArray]] = None,
        gp: GP | None = None,
        factors_init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] = None,
        init_loc: float = 0.0,
        init_scale: float = 0.1,
        init_prob: float = 0.5,
        init_alpha: float = 1.0,
        init_beta: float = 1.0,
        init_shape: float = 10,
        init_rate: float = 10,
    ):
        super().__init__()
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_factors = n_factors

        if isinstance(factor_prior, str):
            factor_prior = dict.fromkeys(self._group_names, factor_prior)

        if isinstance(weight_prior, str):
            weight_prior = dict.fromkeys(self._view_names, weight_prior)

        if isinstance(nonnegative_factors, bool):
            nonnegative_factors = dict.fromkeys(self._group_names, nonnegative_factors)

        if isinstance(nonnegative_weights, bool):
            nonnegative_weights = dict.fromkeys(self._view_names, nonnegative_weights)

        # need to call contiguous() here, otherwise we get a warning from PyTorch:
        # grad and param do not obey the gradient layout contract
        if factors_init_tensor is not None:
            factors_init_tensor = {
                name: {sname: torch.as_tensor(sval).contiguous() for sname, sval in val.items()}
                for name, val in factors_init_tensor.items()
            }

        self._nonnegative_weights = nonnegative_weights
        self._nonnegative_factors = nonnegative_factors
        self._pos_transform = torch.nn.ReLU()

        factor_prior_groups = defaultdict(list)
        for group_name, prior in factor_prior.items():
            factor_prior_groups[prior].append(group_name)
        self._factors = PyroModuleList(
            [
                Prior(
                    prior,
                    names=groups,
                    factor_dim=-3,
                    nonfactor_dim=self._sample_plate_dim,
                    n_factors=n_factors,
                    n_nonfactors=n_samples,
                    gp=gp,
                    init_tensor=factors_init_tensor,
                    init_loc=init_loc,
                    init_scale=init_scale,
                    init_prob=init_prob,
                    init_alpha=init_alpha,
                    init_beta=init_beta,
                    init_shape=init_shape,
                    init_rate=init_rate,
                )
                for prior, groups in factor_prior_groups.items()
            ]
        )

        weight_prior_groups = defaultdict(list)
        for view_name, prior in weight_prior.items():
            weight_prior_groups[prior].append(view_name)
        self._weights = PyroModuleList(
            [
                Prior(
                    prior,
                    names=views,
                    factor_dim=-3,
                    nonfactor_dim=self._feature_plate_dim,
                    n_factors=n_factors,
                    n_nonfactors=n_features,
                    prior_scales=prior_scales,
                    init_loc=init_loc,
                    init_scale=init_scale,
                    init_prob=init_prob,
                    init_alpha=init_alpha,
                    init_beta=init_beta,
                    init_shape=init_shape,
                    init_rate=init_rate,
                )
                for prior, views in weight_prior_groups.items()
            ]
        )

        self._likelihoods = PyroModuleDict(
            {
                view_name: likelihood.pyro_likelihood(
                    view_name=view_name,
                    sample_dim=self._sample_plate_dim,
                    feature_dim=self._feature_plate_dim,
                    sample_means=sample_means,
                    feature_means=feature_means,
                    init_loc=init_loc,
                    init_scale=init_scale,
                    init_prob=init_prob,
                    init_alpha=init_alpha,
                    init_beta=init_beta,
                    init_shape=init_shape,
                    init_rate=init_rate,
                )
                for view_name, likelihood in likelihoods.items()
            }
        )

        self._scale_elbo = True
        n_views = len(self._view_names)
        self._view_scales = dict.fromkeys(self._view_names, 1.0)
        if self._scale_elbo and n_views > 1:
            for view_name, view_n_features in n_features.items():
                self._view_scales[view_name] = (n_views / (n_views - 1)) * (
                    1.0 - view_n_features / sum(n_features.values())
                )

        # guiding variables
        self._guiding_vars_n_categories = guiding_vars_n_categories
        self._guiding_vars_factors = guiding_vars_factors
        self._guiding_vars_scales = guiding_vars_scales

        self._guiding_vars_likelihoods = PyroModuleDict(
            {
                guiding_var_name: PyroLikelihood(
                    guiding_vars_likelihoods[guiding_var_name],
                    view_name=guiding_var_name,
                    sample_dim=self._sample_plate_dim,
                    feature_dim=self._feature_plate_dim,
                    sample_means=sample_means,
                    feature_means={"dummy_name": {guiding_var_name: torch.zeros(1, 1)}},
                )
                for guiding_var_name in self._guiding_vars_names
            }
        )

        self._guiding_locs = PyroParameterDict()
        self._guiding_scales = PyroParameterDict()

        for guiding_var_name in self._guiding_vars_names:
            self._guiding_locs[guiding_var_name] = PyroParam(
                torch.full([self._guiding_vars_n_categories[guiding_var_name], 2], init_loc),
                constraint=constraints.real,
            )
            self._guiding_scales[guiding_var_name] = PyroParam(
                torch.full([self._guiding_vars_n_categories[guiding_var_name], 2], init_scale),
                constraint=constraints.softplus_positive,
            )

    _sample_plate_dim = -2
    _feature_plate_dim = -1

    @property
    def _group_names(self):
        return self._n_samples.keys()

    @property
    def _view_names(self):
        return self._n_features.keys()

    @property
    def _guiding_vars_names(self):
        return self._guiding_vars_factors.keys()

    def _get_plates(self, subsample=None):
        sample_plates = {}

        for group_name in self._group_names:
            sample_plates[group_name] = pyro.plate(
                f"plate_samples_{group_name}",
                self._n_samples[group_name],
                dim=self._sample_plate_dim,
                subsample=subsample[group_name],
            )

        feature_plates = {}
        for view_name in self._view_names:
            feature_plates[view_name] = pyro.plate(
                f"plate_features_{view_name}",
                self._n_features[view_name],
                subsample=torch.arange(  # workaround for https://github.com/pyro-ppl/pyro/pull/3405
                    self._n_features[view_name]
                ),
                dim=self._feature_plate_dim,
            )

        guiding_var_plate = pyro.plate("plate_guiding_vars", 1, subsample=torch.arange(1), dim=self._feature_plate_dim)

        factors_plate = pyro.plate("plate_factors", self._n_factors, dim=-3)

        return sample_plates, feature_plates, guiding_var_plate, factors_plate

    def _model_guiding_vars_weights_normal(self, guiding_var_name, **kwargs):
        weights_dim = self._guiding_vars_n_categories[guiding_var_name]
        return pyro.sample(
            f"guiding_vars_w_{guiding_var_name}",
            dist.Normal(torch.zeros(weights_dim, 2), torch.ones(weights_dim, 2)).to_event(
                2
            ),  # (categories, intercept & slope)
        )

    def _guide_guiding_vars_weights_normal(self, guiding_var_name, **kwargs):
        return pyro.sample(
            f"guiding_vars_w_{guiding_var_name}",
            dist.Normal(self._guiding_locs[guiding_var_name], self._guiding_scales[guiding_var_name]).to_event(2),
        )

    @pyro_method
    def model(self, data, sample_idx, nonmissing_samples, nonmissing_features, covariates, guiding_vars):
        sample_plates, feature_plates, guiding_var_plate, factor_plate = self._get_plates(subsample=sample_idx)

        factors = {}
        for prior in self._factors:
            factors.update(prior.model(factor_plate, sample_plates, covariates=covariates))

        for group_name, group_factors in factors.items():
            if self._nonnegative_factors[group_name]:
                factors[group_name] = self._pos_transform(group_factors)

        weights = {}
        for prior in self._weights:
            weights.update(prior.model(factor_plate, feature_plates))

        for view_name, view_weights in weights.items():
            if self._nonnegative_weights[view_name]:
                weights[view_name] = self._pos_transform(view_weights)

        # sample guiding variable weights
        guided_vars = {}
        for guiding_var_name in self._guiding_vars_names:
            guided_vars[guiding_var_name] = self._model_guiding_vars_weights_normal(guiding_var_name)

        # sample observations
        for group_name, group in data.items():
            gnonmissing_samples = nonmissing_samples[group_name]
            gnonmissing_features = nonmissing_features[group_name]
            for view_name, view_obs in group.items():
                if view_obs.numel() == 0:  # can occur in the last batch of an epoch if the batch is small
                    continue

                vnonmissing_samples = gnonmissing_samples[view_name]
                vnonmissing_features = gnonmissing_features[view_name]

                z = factors[group_name][..., vnonmissing_samples, :]
                w = weights[view_name][..., vnonmissing_features]

                loc = torch.einsum("...ijk,...ikl->...kjl", z, w)

                self._likelihoods[view_name].model(
                    data=view_obs,
                    estimate=loc,
                    group_name=group_name,
                    scale=self._view_scales[view_name],
                    sample_plate=sample_plates[group_name],
                    feature_plate=feature_plates[view_name],
                    nonmissing_samples=vnonmissing_samples,
                    nonmissing_features=vnonmissing_features,
                )

            # guiding variables
            for guiding_var_name in self._guiding_vars_names:
                if group_name not in guiding_vars[guiding_var_name]:
                    continue

                z_guiding = factors[group_name][self._guiding_vars_factors[guiding_var_name], 0]
                w_guiding = guided_vars[guiding_var_name]

                # (n_cats, 1) + (n_cats, 1) * (n_samples,)
                loc = w_guiding[:, 0, None] + w_guiding[:, 1, None] * z_guiding  # (n_cats, n_samples)
                obs_guiding_vars = guiding_vars[guiding_var_name][group_name]

                self._guiding_vars_likelihoods[guiding_var_name].model(
                    data=obs_guiding_vars,
                    estimate=loc,
                    group_name=group_name,
                    scale=self._guiding_vars_scales[guiding_var_name],
                    sample_plate=sample_plates[group_name],
                    feature_plate=guiding_var_plate,
                    nonmissing_samples=slice(None),
                    nonmissing_features=slice(None),
                )

    @pyro_method
    def guide(self, data, sample_idx, nonmissing_samples, nonmissing_features, covariates, guiding_vars):
        sample_plates, feature_plates, guiding_var_plate, factor_plate = self._get_plates(subsample=sample_idx)

        for prior in self._factors:
            prior.guide(factor_plate, sample_plates, covariates=covariates)

        for prior in self._weights:
            prior.guide(factor_plate, feature_plates)

        for guiding_var_name in self._guiding_vars_names:
            self._guide_guiding_vars_weights_normal(guiding_var_name)

        for group_name, group in data.items():
            for view_name in group.keys():
                self._likelihoods[view_name].guide(group_name, sample_plates[group_name], feature_plates[view_name])

            for guiding_var_name in self._guiding_vars_names:
                if group_name in guiding_vars[guiding_var_name]:
                    self._guiding_vars_likelihoods[guiding_var_name].guide(
                        group_name, sample_plates[group_name], guiding_var_plate
                    )

    def get_lr_func(self, base_lr: float, **kwargs):
        modifiers = {}
        for i, prior in enumerate(self._weights):
            modifiers.update(
                {
                    f"{__class__.__name__}._weights.{i}.{pname}": mod
                    for pname, mod in prior.learning_rate_multipliers.items()
                }
            )
        for i, prior in enumerate(self._factors):
            modifiers.update(
                {
                    f"{__class__.__name__}._factors.{i}.{pname}": mod
                    for pname, mod in prior.learning_rate_multipliers.items()
                }
            )

        def lr_func(param_name):
            return dict(lr=base_lr * modifiers.get(param_name, 1), **kwargs)

        return lr_func

    @torch.inference_mode()
    def get_factors(self):
        """Get all factor matrices, z_x."""
        factors = MeanStd({}, {})
        for prior in self._factors:
            for lsidx, vals in enumerate(prior.posterior):
                factors[lsidx].update(vals)

        for group_name in self._group_names:
            if self._nonnegative_factors[group_name]:
                factors.mean[group_name] = self._pos_transform(factors.mean[group_name])
            factors.mean[group_name] = factors.mean[group_name].cpu().numpy()
            factors.std[group_name] = factors.std[group_name].cpu().numpy()

        return factors

    @torch.inference_mode()
    def get_sparse_factor_precisions(self):
        alphas = MeanStd({}, {})
        for prior in self._factors:
            try:
                precisions = prior.posterior_precision
            except AttributeError:
                continue
            for group_name in precisions.shape.keys():
                d = dist.Gamma(concentration=precisions.shape[group_name], rate=precisions.rate[group_name])
                alphas.mean[group_name] = d.mean.cpu().numpy()
                alphas.std[group_name] = d.stddev.cpu().numpy()
        return alphas

    @torch.inference_mode()
    def get_sparse_factor_probabilities(self):
        probs = {}
        for prior in self._factors:
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
        for prior in self._weights:
            for lsidx, vals in enumerate(prior.posterior):
                weights[lsidx].update(vals)

        for view_name in self._view_names:
            if self._nonnegative_weights[view_name]:
                weights.mean[view_name] = self._pos_transform(weights.mean[view_name])
            weights.mean[view_name] = weights.mean[view_name].cpu().numpy()
            weights.std[view_name] = weights.std[view_name].cpu().numpy()

        return weights

    @torch.inference_mode()
    def get_sparse_weight_precisions(self):
        alphas = MeanStd({}, {})
        for prior in self._weights:
            try:
                precisions = prior.posterior_precision
            except AttributeError:
                continue
            for view_name in precisions.shape.keys():
                d = dist.Gamma(concentration=precisions.shape[view_name], rate=precisions.rate[view_name])
                alphas.mean[view_name] = d.mean.cpu().numpy()
                alphas.std[view_name] = d.stddev.cpu().numpy()
        return alphas

    @torch.inference_mode()
    def get_sparse_weight_probabilities(self):
        probs = {}
        for prior in self._weights:
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
        for view_name, likelihood in self._likelihoods.items():
            try:
                disp = likelihood.dispersion
            except AttributeError:
                continue
            dispersion.mean[view_name] = disp.mean
            dispersion.std[view_name] = disp.std

        return dispersion
