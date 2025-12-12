from collections.abc import Mapping, Sequence
from typing import Literal

import pyro
import pyro.distributions as dist
import torch
from numpy.typing import NDArray
from pyro.distributions import constraints
from pyro.nn import PyroModuleList, PyroParam, pyro_method

from ..priors import Prior
from ..pyro.likelihoods import PyroLikelihood
from ..pyro.utils import PyroModuleDict, PyroParameterDict
from ..utils import MeanStd
from .base import Term


class MofaFlex(Term):
    def __init__(
        self,
        n_samples: Mapping[str, int],
        n_features: Mapping[str, int],
        n_factors: int,
        factor_prior: Sequence[Prior],
        weight_prior: Sequence[Prior],
        nonnegative_weights: Mapping[str, bool] | bool = False,
        nonnegative_factors: Mapping[str, bool] | bool = False,
        guiding_vars_likelihoods: Mapping[str, str] | None = None,
        guiding_vars_n_categories: Mapping[str, int] | None = None,
        guiding_vars_factors: Mapping[str, int] | None = None,
        guiding_vars_scales: Mapping[str, float] | None = None,
        feature_means: Mapping[str, Mapping[str, NDArray]] = None,
        sample_means: Mapping[str, Mapping[str, NDArray]] = None,
        factors_init_tensor: Mapping[str, Mapping[Literal["loc", "scale"], NDArray]] = None,
    ):
        super().__init__()
        self._n_samples = n_samples
        self._n_features = n_features
        self._n_factors = n_factors

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

        self._factors = PyroModuleList(
            [
                prior.pyro_prior(
                    factor_dim=-3,
                    nonfactor_dim=self._sample_plate_dim,
                    n_factors=n_factors,
                    n_nonfactors=n_samples,
                    init_tensor=factors_init_tensor,
                )
                for prior in factor_prior
            ]
        )

        self._weights = PyroModuleList(
            [
                prior.pyro_prior(
                    factor_dim=-3, nonfactor_dim=self._feature_plate_dim, n_factors=n_factors, n_nonfactors=n_features
                )
                for prior in weight_prior
            ]
        )

        # guiding variables
        self._guiding_vars_n_categories = guiding_vars_n_categories
        self._guiding_vars_factors = guiding_vars_factors

        total_n_features = 0.1 * sum(self._n_features.values())
        self._guiding_vars_scales = {name: scale * total_n_features for name, scale in guiding_vars_scales.items()}

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

        self._guiding_vars_weights_dims = {}
        for guiding_var_name in self._guiding_vars_names:
            self._guiding_vars_weights_dims[guiding_var_name] = weights_dim = max(
                self._guiding_vars_n_categories[guiding_var_name], 1
            )
            self._guiding_locs[guiding_var_name] = PyroParam(torch.full([weights_dim, 2]), constraint=constraints.real)
            self._guiding_scales[guiding_var_name] = PyroParam(
                torch.full([weights_dim, 2]), constraint=constraints.softplus_positive
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

    def _get_plates(self):
        if len(self._guiding_vars_names):
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

        factors_plate = pyro.plate("plate_factors", self._n_factors, dim=-3)

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
                dist.Normal(
                    self._guiding_locs[guiding_var_name], self._guiding_scales[guiding_var_name]
                ),  # .to_event(2),
            )

    @pyro_method
    def model(
        self,
        data,
        sample_idx,
        sample_plates,
        feature_plates,
        nonmissing_samples,
        nonmissing_features,
        guiding_vars=None,
        **kwargs,
    ):
        guiding_var_plate, guiding_var_coefficients_plate, guiding_var_categories_plates, factor_plate = (
            self._get_plates()
        )

        factors = {}
        for prior in self._factors:
            factors.update(prior.model(factor_plate, sample_plates, **kwargs))

        for group_name, group_factors in factors.items():
            if self._nonnegative_factors[group_name]:
                factors[group_name] = self._pos_transform(group_factors)

        weights = {}
        for prior in self._weights:
            weights.update(prior.model(factor_plate, feature_plates))

        for view_name, view_weights in weights.items():
            if self._nonnegative_weights[view_name]:
                weights[view_name] = self._pos_transform(view_weights)

        estimates = {}
        for group_name, group in data.items():
            gestimates = {}
            gnonmissing_samples = nonmissing_samples[group_name]
            gnonmissing_features = nonmissing_features[group_name]
            for view_name, view_obs in group.items():
                if view_obs.numel() == 0:  # can occur in the last batch of an epoch if the batch is small
                    continue

                vnonmissing_samples = gnonmissing_samples[view_name]
                vnonmissing_features = gnonmissing_features[view_name]

                z = factors[group_name][..., vnonmissing_samples, :]
                w = weights[view_name][..., vnonmissing_features]

                gestimates[view_name] = torch.einsum("...ijk,...ikl->...kjl", z, w)
            estimates[group_name] = gestimates

        for guiding_var_name, guiding_var_factor_idx in self._guiding_vars_factors.items():
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

                self._guiding_vars_likelihoods[guiding_var_name].model(
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
        data,
        sample_idx,
        sample_plates,
        feature_plates,
        nonmissing_samples,
        nonmissing_features,
        guiding_vars=None,
        **kwargs,
    ):
        (guiding_var_plate, guiding_var_coefficients_plate, guiding_var_categories_plates, factor_plate) = (
            self._get_plates()
        )

        for prior in self._factors:
            prior.guide(factor_plate, sample_plates, **kwargs)

        for prior in self._weights:
            prior.guide(factor_plate, feature_plates)

        if len(self._guiding_vars_factors) > 0:
            for guiding_var_name, guiding_var in guiding_vars.items():
                self._guide_guiding_vars_weights_normal(
                    guiding_var_name, guiding_var_coefficients_plate, guiding_var_categories_plates
                )
                for group_name in guiding_var.keys():
                    self._guiding_vars_likelihoods[guiding_var_name].guide(
                        group_name, sample_plates[group_name], guiding_var_plate
                    )

    @property
    def learning_rate_multipliers(self):
        for i, prior in enumerate(self._weights):
            yield from ((f"_weights.{i}.{pname}", mod) for pname, mod in prior.learning_rate_multipliers)
        for i, prior in enumerate(self._factors):
            yield from ((f"_factors.{i}.{pname}", mod) for pname, mod in prior.learning_rate_multipliers)

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
            factors.mean[group_name] = factors.mean[group_name].cpu().numpy().T
            factors.std[group_name] = factors.std[group_name].cpu().numpy().T

        return factors

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
            weights.mean[view_name] = weights.mean[view_name].cpu().numpy().T
            weights.std[view_name] = weights.std[view_name].cpu().numpy().T

        return weights

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
