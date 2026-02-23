from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from inspect import isabstract, signature
from itertools import islice

import numpy as np
import pyro
import torch
from pyro import distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample, pyro_method

from ...utils import MeanStd, PyroMeta, checked_baseclass


@checked_baseclass(
    required_init_args=("view_name", "sample_dim", "feature_dim", "nsamples", "nfeatures"), registry="dict"
)
class Likelihood(ABC, PyroModule, metaclass=PyroMeta):
    """Base class for MOFA-FLEX likelihoods used in the Pyro model.

    Subclasses must implement `_model`, which returns a Pyro distribution object to be used as likelihood,
    and `_guide`, which implements the variational distribution. Its return value is ignored.

    Args:
        view_name: The view (or guiding variable) name.
        sample_dim: The sample dimension.
        feature_dim: The feature dimension.
    """

    def __init__(
        self, view_name: str, sample_dim: int, feature_dim: int, nsamples: Mapping[str, int], nfeatures: int, **kwargs
    ):
        super().__init__(**kwargs)
        self._view_name = view_name
        self._sample_dim = sample_dim
        self._feature_dim = feature_dim
        self._nsamples = nsamples
        self._nfeatures = nfeatures

        self._mode = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not isabstract(cls) and cls.__name__[0] != "_":
            init_sig = signature(cls.__init__)
            for i, (arg, param) in enumerate(
                zip(
                    ("view_name", "sample_dim", "feature_dim", "nsamples", "nfeatures"),
                    islice(init_sig.parameters.values(), 1, None),
                    strict=False,
                )
            ):
                if arg != param.name:
                    raise TypeError(f"Constructor of class {cls} is missing the {arg} argument at position {i + 1}.")

    @pyro_method
    def model(
        self,
        id: str,
        data: torch.Tensor,
        estimate: torch.Tensor,
        group_name: str,
        scale: float,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ):
        """Pyro model for the likelihood.

        Args:
            id: ID to be used in Pyro sample site names to make them unique if multiple instances of the same likelihood are used.
            data: The observed data.
            estimate: The model estimate.
            group_name: The group name.
            scale: Scale for the likelihood of this view.
            sample_plate: Pyro plate for the samples.
            feature_plate: Pyro plate for the features.
            nonmissing_samples: Index tensor indicating which global sample indices of the current minibatch are present
                in the current group and view.
            nonmissing_features: Index tensor indicating which global feature indices of the current minibatch are present
                in the current group and view.
        """
        self._mode = "model"

        data_mask = ~torch.isnan(data)
        data = torch.nan_to_num(data, nan=0.0)

        nonmissing_sample_plate = pyro.plate(
            f"{id}_samples_{group_name}_{self._view_name}",
            sample_plate.size,
            dim=sample_plate.dim,
            subsample=sample_plate.indices[nonmissing_samples],
        )

        nonmissing_feature_plate = pyro.plate(
            f"{id}_features_{group_name}_{self._view_name}",
            feature_plate.size,
            dim=feature_plate.dim,
            subsample=(
                feature_plate.indices[nonmissing_features] if isinstance(nonmissing_features, torch.Tensor) else None
            ),
        )  # pyro.plate can't handle slices
        obsdist = self._model(
            id, estimate, group_name, sample_plate, feature_plate, nonmissing_samples, nonmissing_features
        )
        with (
            pyro.poutine.mask(mask=data_mask),
            pyro.poutine.scale(scale=scale),
            nonmissing_sample_plate,
            nonmissing_feature_plate,
        ):
            return pyro.sample(f"{id}_observed_{group_name}_{self._view_name}", obsdist, obs=data)

    @abstractmethod
    def _model(
        self,
        id: str,
        estimate: torch.Tensor,
        group_name: str,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ) -> pyro.distributions.Distribution:
        """Pyro model for the likelihood.

        Args:
            id: ID to be used in Pyro sample site names to make them unique if multiple instances of the same likelihood are used.
            estimate: The model estimate.
            group_name: The group name.
            sample_plate: Pyro plate for the samples.
            feature_plate: Pyro plate for the features.
            nonmissing_samples: Index tensor indicating which global sample indices of the current minibatch are present
                in the current group and view.
            nonmissing_features: Index tensor indicating which global feature indices of the current minibatch are present
                in the current group and view.

        Returns:
            A Pyro distribution object that can be used with `pyro.sample`.
        """
        pass

    def guide(self, id: str, group_name: str, sample_plate: pyro.plate, feature_plate: pyro.plate):
        """Pyro guide for the likelhood.

        Args:
            id: ID to be used in Pyro sample site names to make them unique if multiple instances of the same likelihood are used.
            group_name: The group name.
            sample_plate: Pyro plate for the samples.
            feature_plate: Pyro plate for the features.
        """
        self._mode = "guide"
        return self._guide(id, group_name, sample_plate, feature_plate)

    @abstractmethod
    def _guide(self, id: str, group_name: str, sample_plate: pyro.plate, feature_plate: pyro.plate):
        """Pyro guide for the likelhood.

        Args:
            id: ID to be used in Pyro sample site names to make them unique if multiple instances of the same likelihood are used.
            group_name: The group name.
            sample_plate: Pyro plate for the samples.
            feature_plate: Pyro plate for the features.
        """
        pass

    def _random_attr(
        self,
        generative_dist: pyro.distributions.Distribution | Callable,
        variational_dist: pyro.distributions.Distribution | Callable,
    ) -> PyroSample:
        """Helper method to prepare a `PyroSample` attribute.

        This is useful for random variables that are common to all groups of a view, since the `PyroLikelihood` object will be called
        for each group separately, and PyroSample objects cache their sampled values. The helper is needed to make the `PyroSample`
        behave correctly for both model and guide.

        Args:
            generative_dist: The generative distribution.
            variational_dist: The variational distribution.
        """

        def dist(self):
            if self._mode == "model":
                cdist = generative_dist
            else:
                cdist = variational_dist
            if not hasattr(cdist, "sample"):
                cdist = cdist(self)  # double indirection for lazy access to PyroParam attributes
            return cdist

        return PyroSample(dist)


class LikelihoodWithDispersion(Likelihood):
    """Base class for Pyro likelihoods with a dispersion parameter."""

    def __init__(
        self,
        view_name: str,
        sample_dim: int,
        feature_dim: int,
        nsamples: dict[str, int],
        nfeatures: int,
        *,
        init_loc: float = 1.0,
        init_scale: float = 0.1,
        **kwargs,
    ):
        super().__init__(view_name, sample_dim, feature_dim, nsamples, nfeatures, **kwargs)

        shape = self._nfeatures, *((1,) * (abs(self._feature_dim) - 1))
        self._loc = PyroParam(torch.full(size=shape, fill_value=np.log(init_loc) - 0.5 * init_scale**2))
        self._scale = PyroParam(
            torch.full(size=shape, fill_value=init_scale), constraint=dist.constraints.softplus_positive
        )
        self._dispersion = self._random_attr(
            dist.Gamma(1e-3, 1e-3), lambda self: dist.LogNormal(self._loc, self._scale)
        )

    @pyro_method
    def _model_dispersion(
        self,
        id: str,
        estimate: torch.Tensor,
        group_name: str,
        sample_plate: pyro.plate,
        feature_plate: pyro.plate,
        nonmissing_samples: torch.Tensor | slice,
        nonmissing_features: torch.Tensor | slice,
    ) -> torch.Tensor:
        """Pyro model for the dispersion.

        Args:
            id: ID to be used in Pyro sample site names to make them unique if multiple instances of the same likelihood are used.
            estimate: The model estimate.
            group_name: The group name.
            scale: Scale for the likelihood of this view.
            sample_plate: Pyro plate for the samples.
            feature_plate: Pyro plate for the features.
            nonmissing_samples: Index tensor indicating which global sample indices of the current minibatch are present
                in the current group and view.
            nonmissing_features: Index tensor indicating which global feature indices of the current minibatch are present
                in the current group and view.

        Returns:
            A tensor with dispersion values.
        """
        with feature_plate:
            dispersion = self._dispersion
        return dispersion.movedim(self._feature_dim, 0)[nonmissing_features, ...].movedim(0, self._feature_dim)

    @pyro_method
    def _guide(self, id: str, group_name: str, sample_plate: pyro.plate, feature_plate: pyro.plate):
        """Pyro guide for the dispersion."""
        with feature_plate:
            return self._dispersion

    @property
    @torch.inference_mode()
    def dispersion(self) -> MeanStd:
        """The estimated dispersion."""
        squeezedims = list(range(self._loc.ndim))
        del squeezedims[self._feature_dim]

        # TODO: use actual mean and std of LogNormal
        return MeanStd(self._loc.squeeze(squeezedims).cpu().numpy(), self._scale.squeeze(squeezedims).cpu().numpy())
