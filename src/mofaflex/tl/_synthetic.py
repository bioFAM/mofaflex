import itertools
import logging
import math
from collections.abc import Iterable, Sequence
from typing import Literal

import anndata as ad
import mudata as md
import numpy as np
import numpy.typing as npt

_logger = logging.getLogger(__name__)


class DataGenerator:
    """Generator class for creating synthetic multi-view data with latent factors.

    This class generates synthetic data with specified properties including shared and private
    latent factors, different likelihoods, and optional covariates and response variables.

    Attributes:
        n_features: List of feature counts for each view.
        n_samples: Number of samples to generate.
        n_views: Number of views in the dataset.
        n_fully_shared_factors: Number of factors shared across all views.
        n_partially_shared_factors: Number of factors shared between some views.
        n_private_factors: Number of factors unique to individual views.
        n_covariates: Number of observed covariates.
        likelihoods: List of likelihood types for each view.
        factor_size_params: Parameters for factor size distribution.
        factor_size_dist: Type of distribution for factor sizes.
        n_active_factors: Number or fraction of active factors.
        nmf: List indicating which views should use non-negative matrix factorization.
    """

    def __init__(
        self,
        n_features: Sequence[int],
        n_samples: int = 1000,
        likelihoods: Sequence[Literal["Normal", "Bernoulli", "Poisson"]] | None = None,
        n_fully_shared_factors: int = 2,
        n_partially_shared_factors: int = 15,
        n_private_factors: int = 3,
        factor_size_params: tuple[float, float] | None = None,
        factor_size_dist: Literal["Uniform", "Gamma"] = "Uniform",
        n_active_factors: float = 1.0,
        n_response: int = 0,
        nmf: Sequence[bool] | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_views = len(self.n_features)
        self.n_fully_shared_factors = n_fully_shared_factors
        self.n_partially_shared_factors = n_partially_shared_factors
        self.n_private_factors = n_private_factors

        if factor_size_params is None:
            if factor_size_dist == "Uniform":
                _logger.info(
                    "Using a uniform distribution with parameters 0.05 and 0.15 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (0.05, 0.15)
            elif factor_size_dist == "Gamma":
                _logger.info(
                    "Using a gamma distribution with shape of 1 and scale of 50 "
                    "for generating the number of active factor loadings."
                )
                factor_size_params = (1.0, 50.0)

        if isinstance(factor_size_params, tuple):
            factor_size_params = [factor_size_params for _ in range(self.n_views)]

        self.factor_size_params = factor_size_params
        self.factor_size_dist = factor_size_dist

        # custom assignment
        if likelihoods is None:
            likelihoods = ["Normal" for _ in range(self.n_views)]
        self.likelihoods = likelihoods

        self.n_active_factors = n_active_factors

        if nmf is None:
            nmf = [False for _ in range(self.n_views)]
        self.nmf = nmf

        # set upon data generation
        # latent factors
        self._z = None
        # factor loadings
        self._ws = None
        self._sigmas = None
        self._ys = None
        self._w_masks = None
        self._noisy_w_masks = None
        self._active_factor_indices = None
        self._view_factor_mask = None
        # set when introducing missingness
        self._presence_masks = None

    @property
    def n_factors(self) -> int:
        """Total number of factors."""
        return self.n_fully_shared_factors + self.n_partially_shared_factors + self.n_private_factors

    def _to_matrix(self, matrix_list):
        return np.concatenate(matrix_list, axis=1)

    def _attr_to_matrix(self, attr_name):
        attr = getattr(self, attr_name)
        if attr is not None:
            attr = self._to_matrix(attr)
        return attr

    def _mask_to_nan(self):
        nan_masks = []
        for mask in self._presence_masks:
            nan_mask = np.array(mask, dtype=np.float32, copy=True)
            nan_mask[nan_mask == 0] = np.nan
            nan_masks.append(nan_mask)
        return nan_masks

    def _mask_to_bool(self):
        bool_masks = []
        for mask in self._presence_masks:
            bool_mask = mask == 1.0
            bool_masks.append(bool_mask)
        return bool_masks

    def _missing_ys(self, log=True) -> list[npt.NDArray[np.float32]]:
        if self._ys is None:
            if log:
                _logger.warning("Generate data first by calling `generate`.")
            return []
        if self._presence_masks is None:
            if log:
                _logger.warning("Introduce missing data first by calling `generate_missingness`.")
            return self._ys

        nan_masks = self._mask_to_nan()
        return [self._ys[m] * nan_masks[m] for m in range(self.n_views)]

    @property
    def missing_y(self) -> npt.NDArray[np.float32]:
        """Generated data with non-missing values replaced with `np.nan`."""
        return self._to_matrix(self._missing_ys())

    @property
    def y(self) -> npt.NDArray[np.float32]:
        """Generated data."""
        return self._attr_to_matrix("_ys")

    @property
    def w(self) -> npt.NDArray[np.float32]:
        """Generated weights."""
        return self._attr_to_matrix("_ws")

    @property
    def z(self) -> npt.NDArray[np.float32]:
        """Generated latent factors."""
        return self._z

    @property
    def w_mask(self) -> npt.NDArray[np.bool_]:
        """Gene set mask describing co-expressed genes."""
        return self._attr_to_matrix("_w_masks")

    @property
    def noisy_w_mask(self) -> npt.NDArray[np.bool_]:
        """Gene set mask describing co-expressed genes, with added noise."""
        return self._attr_to_matrix("_noisy_w_masks")

    def _generate_view_factor_mask(self, rng=None, all_combs=False):
        if all_combs and self.n_views == 1:
            _logger.warning("Single view dataset, cannot generate factor combinations for a single view.")
            all_combs = False
        if all_combs:
            _logger.warning(f"Generating all possible binary combinations of {self.n_views} variables.")
            self.n_fully_shared_factors = 1
            self.n_private_factors = self.n_views
            self.n_partially_shared_factors = 2**self.n_views - 2 - self.n_private_factors
            _logger.warning(
                "New factor configuration: "
                f"{self.n_fully_shared_factors} fully shared, "
                f"{self.n_partially_shared_factors} partially shared, "
                f"{self.n_private_factors} private factors."
            )

            return np.array([list(i) for i in itertools.product([1, 0], repeat=self.n_views)], dtype=bool)[:-1, :].T
        if rng is None:
            rng = np.random.default_rng()

        view_factor_mask = np.ones([self.n_views, self.n_factors], dtype=bool)

        for factor_idx in range(self.n_fully_shared_factors, self.n_factors):
            # exclude view subsets for partially shared factors
            if factor_idx < self.n_fully_shared_factors + self.n_partially_shared_factors:
                if self.n_views > 2:
                    exclude_view_subset_size = rng.integers(1, self.n_views - 1)
                else:
                    exclude_view_subset_size = 0

                exclude_view_subset = rng.choice(self.n_views, exclude_view_subset_size, replace=False)
            # exclude all but one view for private factors
            else:
                include_view_idx = rng.integers(self.n_views)
                exclude_view_subset = [i for i in range(self.n_views) if i != include_view_idx]

            for m in exclude_view_subset:
                view_factor_mask[m, factor_idx] = 0

        if self.n_private_factors >= self.n_views:
            view_factor_mask[-self.n_views :, -self.n_views :] = np.eye(self.n_views)

        return view_factor_mask

    def normalize(self, with_std: bool = False):
        """Normalize data with a Gaussian likelihood to zero mean and optionally unit variance.

        Args:
            with_std: If `True`, also normalize to unit variance. Otherwise, only shift to zero mean.
        """
        for m in range(self.n_views):
            if self.likelihoods[m] == "Normal":
                y = self._ys[m]
                y -= y.mean(axis=0)
                if with_std:
                    y_std = y.std(axis=0)
                    y = np.divide(y, y_std, out=np.zeros_like(y), where=y_std != 0)
                self._ys[m] = y

    def _sigmoid(self, x: float):
        return 1.0 / (1 + np.exp(-x))

    def generate(
        self, rng: np.random.Generator = np.random.default_rng(), all_combs: bool = False, overwrite: bool = False
    ):
        """Generate synthetic data.

        Args:
            rng: The random number generator.
            all_combs: Wether to generate all combinations of active factors and views. If `True`, the model
                will have 1 shared factor, `n_views` private factors, and `2**n_views - n_views - 2` partially
                shared factors.
            overwrite: Whether to overwrite already generated data
        """
        if self._ys is not None and not overwrite:
            raise ValueError("Data has already been generated, to generate new data please set `overwrite` to True.")

        view_factor_mask = self._generate_view_factor_mask(rng, all_combs)

        n_active_factors = self.n_active_factors
        if n_active_factors <= 1.0:
            # if fraction of active factors convert to int
            n_active_factors = int(n_active_factors * self.n_factors)

        active_factor_indices = rng.choice(self.n_factors, size=math.ceil(n_active_factors), replace=False)
        active_factor_indices.sort()

        # generate factor scores which lie in the latent space
        z = rng.standard_normal((self.n_samples, self.n_factors))

        if any(self.nmf):
            z = np.abs(z)

        ws = []
        sigmas = []
        ys = []
        w_masks = []

        for factor_idx in range(self.n_factors):
            if factor_idx not in active_factor_indices:
                view_factor_mask[:, factor_idx] = 0

        for m in range(self.n_views):
            n_features = self.n_features[m]
            w_shape = (self.n_factors, n_features)
            w = rng.standard_normal(w_shape)
            w_mask = np.zeros(w_shape, dtype=np.bool_)

            fraction_active_features = {
                "Gamma": (
                    lambda shape, scale, n_features=n_features: (
                        (rng.gamma(shape, scale, self.n_factors) + 20) / n_features
                    )
                ),
                "Uniform": lambda low, high, n_features=n_features: rng.uniform(low, high, self.n_factors),
            }[self.factor_size_dist](self.factor_size_params[m][0], self.factor_size_params[m][1])

            for factor_idx, faft in enumerate(fraction_active_features):
                if view_factor_mask[m, factor_idx]:
                    w_mask[factor_idx] = rng.choice(2, n_features, p=[1 - faft, faft])

            # set small values to zero
            tiny_w_threshold = 0.1
            w_mask[np.abs(w) < tiny_w_threshold] = False
            # add some noise to avoid exactly zero values
            w = np.where(w_mask, w, rng.standard_normal(w_shape) / 100)
            assert ((np.abs(w) > tiny_w_threshold) == w_mask).all()

            if self.nmf[m]:
                w = np.abs(w)

            y_loc = np.matmul(z, w)

            # generate feature sigmas
            sigma = 1.0 / np.sqrt(rng.gamma(10.0, 1.0, n_features))

            match self.likelihoods[m]:
                case "Normal":
                    y = rng.normal(loc=y_loc, scale=sigma)
                    if self.nmf[m]:
                        y = np.abs(y)
                case "Bernoulli":
                    y = rng.binomial(1, self._sigmoid(y_loc))
                case "Poisson":
                    rate = np.exp(y_loc)
                    y = rng.poisson(rate)

            ws.append(w)
            sigmas.append(sigma)
            ys.append(y)
            w_masks.append(w_mask)

        self._z = z
        self._ws = ws
        self._w_masks = w_masks
        self._sigmas = sigmas
        self._ys = ys
        self._active_factor_indices = active_factor_indices
        self._view_factor_mask = view_factor_mask

    def get_noisy_mask(
        self,
        rng: np.random.Generator = np.random.default_rng(),
        noise_fraction: float = 0.1,
        informed_view_indices: Iterable[int] | None = None,
    ) -> list[npt.NDArray[bool]]:
        """Generate a noisy version of `w_mask`, the mask describing co-expressed genes.

        Noisy in this context means that some annotations are wrong, i.e. some genes active in a particular factor
        are marked as inactive, and some genes inactive in a factor are marked as active.

        Args:
            rng: The random number generator.
            noise_fraction: Fraction of active genes per factor that will be marked as inactive. The same number of
                inactive genes will be marked as active.
            informed_view_indices: Indices of views that will be used to benchmark informed models. Noisy masks will
                be generated only for those views. For uninformed views, th enoisy masks will be filled with `False`.

        Returns:
            A list with a noisy mask for each view.
        """
        if informed_view_indices is None:
            _logger.warning("Parameter `informed_view_indices` set to None, adding noise to all views.")
            informed_view_indices = list(range(self.n_views))

        noisy_w_masks = [mask.copy() for mask in self._w_masks]

        if len(informed_view_indices) == 0:
            _logger.warning(
                "Parameter `informed_view_indices` set to an empty list, removing information from all views."
            )
            self._noisy_w_masks = [np.ones_like(mask) for mask in noisy_w_masks]
            return self._noisy_w_masks

        for m in range(self.n_views):
            noisy_w_mask = noisy_w_masks[m]

            if m in informed_view_indices:
                fraction_active_cells = noisy_w_mask.mean(axis=1).sum() / self._view_factor_mask[0].sum()
                for factor_idx in range(self.n_factors):
                    active_cell_indices = noisy_w_mask[factor_idx, :].nonzero()[0]
                    # if all features turned off
                    # => simulate random noise in terms of false positives only
                    if len(active_cell_indices) == 0:
                        _logger.warning(
                            f"Factor {factor_idx} is completely off, inserting "
                            f"{(100 * fraction_active_cells):.2f}%% false positives."
                        )
                        active_cell_indices = rng.choice(
                            self.n_features[m], int(self.n_features[m] * fraction_active_cells), replace=False
                        )

                    inactive_cell_indices = (noisy_w_mask[factor_idx, :] == 0).nonzero()[0]
                    n_noisy_cells = int(noise_fraction * len(active_cell_indices))
                    swapped_indices = zip(
                        rng.choice(len(active_cell_indices), n_noisy_cells, replace=False),
                        rng.choice(len(inactive_cell_indices), n_noisy_cells, replace=False),
                        strict=False,
                    )

                    for on_idx, off_idx in swapped_indices:
                        noisy_w_mask[factor_idx, active_cell_indices[on_idx]] = False
                        noisy_w_mask[factor_idx, inactive_cell_indices[off_idx]] = True

            else:
                noisy_w_mask.fill(False)

        self._noisy_w_masks = noisy_w_masks
        return self._noisy_w_masks

    def generate_missingness(
        self,
        rng: np.random.Generator = np.random.default_rng(),
        n_partial_samples: int = 0,
        n_partial_features: int = 0,
        missing_fraction_partial_features: float = 0.0,
        random_fraction: float = 0.0,
    ):
        """Mark observations as missing.

        Args:
            rng: The random number generator.
            n_partial_samples: Number of samples marked as missing in at least one random view. If the model has only
                one view, this has no effect.
            n_partial_features: Number of features marked as missing in some samples.
            missing_fraction_partial_features: Fraction of samples marked as missing due to `n_partial_features`.
            random_fraction: Fraction of all observations marked as missing at random.
        """
        sample_view_mask = np.ones((self.n_samples, self.n_views), dtype=np.bool_)
        missing_sample_indices = rng.choice(self.n_samples, n_partial_samples, replace=False)

        # partially missing samples
        for ms_idx in missing_sample_indices:
            if self.n_views > 1:
                exclude_view_subset_size = rng.integers(1, self.n_views)
            else:
                exclude_view_subset_size = 0
            exclude_view_subset = rng.choice(self.n_views, exclude_view_subset_size, replace=False)
            sample_view_mask[ms_idx, exclude_view_subset] = 0

        mask = np.repeat(sample_view_mask, self.n_features, axis=1)

        # partially missing features
        missing_feature_indices = rng.choice(sum(self.n_features), n_partial_features, replace=False)

        for mf_idx in missing_feature_indices:
            random_sample_indices = rng.choice(
                self.n_samples, int(self.n_samples * missing_fraction_partial_features), replace=False
            )
            mask[random_sample_indices, mf_idx] = 0

        # remove random fraction
        mask[
            np.unravel_index(
                rng.choice(np.prod(mask.shape), size=int(random_fraction * np.prod(mask.shape)), replace=False),
                mask.shape,
            )
        ] = False

        view_feature_offsets = [0, *np.cumsum(self.n_features).tolist()]
        masks = []
        for offset_idx in range(len(view_feature_offsets) - 1):
            start_offset = view_feature_offsets[offset_idx]
            end_offset = view_feature_offsets[offset_idx + 1]
            masks.append(mask[:, start_offset:end_offset])

        self._presence_masks = masks

    def _permute_features(self, lst, new_order):
        return [np.take(arr, o, axis=-1) for arr, o in zip(lst, new_order, strict=False)]

    def _permute_factors(self, lst, new_order):
        return [arr[new_order, :] for arr in lst]

    def permute_features(self, new_feature_order: Sequence[Iterable[int]]):
        """Permute features.

        Args:
            new_feature_order: New ordering of features.
        """
        if len(new_feature_order) != len(self.n_features) or any(
            len(new) != old for new, old in zip(new_feature_order, self.n_features, strict=False)
        ):
            raise ValueError("Length of new order list must equal the number of features.")
        if (
            any(np.min(new) != 0 for new in new_feature_order)
            or any(np.max(new) != old - 1 for new, old in zip(new_feature_order, self.n_features, strict=False))
            or any(np.unique(new).size != len(new) for new in new_feature_order)
        ):
            raise ValueError(f"New order must contain all integers in [0, {self.n_features}).")

        self._ws = self._permute_features(self._ws, new_feature_order)
        self._w_masks = self._permute_features(self._w_masks, new_feature_order)
        if self._noisy_w_masks is not None:
            self._noisy_w_masks = self._permute_features(self._noisy_w_masks, new_feature_order)
        self._sigmas = self._permute_features(self._sigmas, new_feature_order)
        self._ys = self._permute_features(self._ys, new_feature_order)
        if self._presence_masks is not None:
            self._presence_masks = self._permute_features(self._presence_masks, new_feature_order)

    def permute_factors(self, new_factor_order: Iterable[int]):
        """Permute factors.

        Args:
            new_factor_order: New ordering of factors.
        """
        if len(new_factor_order) != self.n_factors:
            raise ValueError("Length of new order list must equal the number of factors.")
        new_factor_order = np.asarray(new_factor_order)
        if (
            new_factor_order.min() != 0
            or new_factor_order.max() != self.n_factors - 1
            or np.unique(new_factor_order).size != new_factor_order.size
        ):
            raise ValueError(f"New order must contain all integers in [0, {self.n_factors}).")

        self._z = self._z[:, np.array(new_factor_order)]
        self._ws = self._permute_factors(self._ws, new_factor_order)
        self._w_masks = self._permute_factors(self._w_masks, new_factor_order)
        if self._noisy_w_masks is not None:
            self._noisy_w_masks = self._permute_factors(self._noisy_w_masks, new_factor_order)
        self._view_factor_mask = [self._view_factor_mask[m, np.array(new_factor_order)] for m in range(self.n_views)]
        self._active_factor_indices = np.nonzero(np.isin(new_factor_order, self._active_factor_indices))[0]

    def to_mudata(self, noisy=False) -> md.MuData:
        """Export the generated data as a `MuData` object.

        The `AnnData` objects generated for each view will have their weights in `.varm["w"]` and the gene set mask
        in `.varm["w_mask"]. The latent factors will be in `.obsm["z"]` of the `MuData` object, the likelihoods in
        `.uns["likelihoods"]` and the number of active factors in `.uns["n_active_factors"]`.

        Args:
            noisy: Whether to export the noisy or noise-free gene set mask.
        """
        view_names = []
        ad_dict = {}
        ys = self._missing_ys(log=False)
        for m in range(self.n_views):
            adata = ad.AnnData(ys[m].astype(np.float32, copy=False))
            adata.var_names = f"feature_group_{m}:" + adata.var_names
            adata.varm["w"] = self._ws[m].T
            w_mask = self._w_masks[m].T
            if noisy:
                w_mask = self._noisy_w_masks[m].T
            adata.varm["w_mask"] = w_mask
            view_name = f"feature_group_{m}"
            ad_dict[view_name] = adata
            view_names.append(view_name)

        mdata = md.MuData(ad_dict)
        mdata.uns["likelihoods"] = dict(zip(view_names, self.likelihoods, strict=False))
        mdata.uns["n_active_factors"] = self.n_active_factors
        mdata.obsm["z"] = self._z

        return mdata
