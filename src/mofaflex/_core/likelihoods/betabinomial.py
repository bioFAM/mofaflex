import logging
from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from array_api_compat import array_namespace
from scipy.sparse import issparse
from scipy.special import expit, logit
from scipy.stats import betabinom

from ..datasets import MofaFlexDataset
from ..settings import settings
from ..utils import Matrix, Vector, nanmean
from .base import R2, Likelihood, LogLikelihoods
from .pyro import BetaBinomial as PyroBetaBinomial
from .pyro import Likelihood as PyroLikelihood

_logger = logging.getLogger(__name__)


class BetaBinomial(Likelihood):
    r"""Beta-binomial likelihood.

    Requires the data to have layers named "sample" and "total".
    """

    _priority = 10
    _state_attrs = ("_shift", "_dispersion")

    @staticmethod
    def _calc_shift(adata: AnnData):
        shift = logit(nanmean(adata.layers["sample"] / adata.layers["total"], axis=0))
        shift[~np.isfinite(shift)] = 0
        return shift

    def __init__(self, view_name: str, data: MofaFlexDataset, nonnegative: bool):
        super().__init__(view_name, data, nonnegative)
        self._shift = data.apply_to_view(
            view_name,
            lambda adata, group_name: align_local_array_to_global(  # noqa: F821
                self._calc_shift(adata), group_name, self._view_name, align_to="features"
            ),
        )
        self._dispersion = None

        if nonnegative:
            _logger.warning(
                f"The {self.__class__.__name__} likelihood does not support nonnegative views. Your results may be suboptimal."
            )

    def _get_pyro_likelihood(self, data: MofaFlexDataset, sample_dim: int, feature_dim: int) -> PyroLikelihood:
        return PyroBetaBinomial(
            self._view_name,
            sample_dim,
            feature_dim,
            data.n_samples,
            data.n_features[self._view_name],
            shift=self._shift,
        )

    def on_train_end(self, *args, **kwargs):
        self._dispersion = self._pyro_likelihood.dispersion
        self._shift = self._impute_feature_summary_statistics(self._shift)

    @classmethod
    def _validate(cls, adata: AnnData, view_name: str) -> str | None:
        if not {"sample", "total"} <= adata.layers.keys():
            return f"The {cls.__name__} likelihood requires two layers named 'sample' and 'total', got {tuple(adata.layers.keys())} for view {view_name}."
        sample, total = adata.layers["sample"], adata.layers["total"]
        valid = sample <= total
        if issparse(valid):
            valid = valid.data
        xp = array_namespace(valid)
        valid = valid[~xp.isnan(valid)]
        if not xp.all(valid):
            return (
                f"Some values in the 'sample' layer are greater than values in the 'total' layer in view {view_name}."
            )

        if issparse(sample):
            sample = sample.data
        xp = array_namespace(sample)
        sample = sample[~xp.isnan(sample)]
        if not xp.allclose(sample, xp.round(sample)) or sample.min() < 0:
            return f"The {cls.__name__} likelihood in view {view_name} must be used with count (non-negative integer) data (got non-count data for the 'sample' layer)."

        if issparse(total):
            total = total.data
        xp = array_namespace(total)
        total = total[~xp.isnan(total)]
        if not xp.allclose(total, xp.round(total)) or total.min() < 0:
            return f"The {cls.__name__} likelihood in view {view_name} must be used with count (non-negative integer) data (got non-count data for the 'total' layer)."

        return None

    def _logpmf(
        self,
        y_true: Mapping[str | None, Matrix[np.number]],
        mean: Matrix[np.floating],
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating]:
        dispersion = self._dispersion.mean[feature_idx]
        return betabinom.logpmf(
            k=y_true["sample"],
            n=y_true["total"],
            a=dispersion * mean + settings.eps,
            b=dispersion * (1 - mean) + settings.eps,
        )

    def _r2_impl(
        self,
        y_true: Mapping[str | None, Matrix[np.number]],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> R2:
        dispersion = self._dispersion.mean[feature_idx]

        nu1 = (1 - y_pred) * (dispersion + y_true["total"]) / (dispersion + 1)
        ss_res = np.nansum(self._dV(y_true["sample"], y_pred * y_true["total"], nu1))
        ss_tot = np.nansum(
            self._dV(y_true["sample"], expit(self._shift[group_name][feature_idx]) * y_true["total"], nu1)
        )
        return R2(ss_res, ss_tot)

    def _deviance_explained_impl(
        self,
        y_true: Mapping[str | None, Matrix[np.number]],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> LogLikelihoods:
        # use the estimated dispersion for all three models, otherwise the log-likelihoods are not comparable
        loglik_saturated = np.nansum(
            self._logpmf(
                y_true, np.clip(y_true["sample"] / y_true["total"], settings.eps, 1 - settings.eps), feature_idx
            )
        )
        loglik_null = np.nansum(self._logpmf(y_true, expit(self._shift[group_name][feature_idx]), feature_idx))
        loglik_model = np.nansum(
            self._logpmf(y_true, self.transform_prediction(y_pred, group_name, sample_idx, feature_idx), feature_idx)
        )

        return LogLikelihoods(loglik_saturated, loglik_null, loglik_model)

    def transform_prediction(
        self,
        prediction: Matrix[np.floating] | Vector[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating] | Vector[np.floating]:
        return expit(prediction + self._shift[group_name][feature_idx])

    def transform_data(
        self,
        data: Mapping[str | None, Matrix[np.number]],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating]:
        return (
            logit(np.clip(data["sample"] / data["total"], settings.eps, 1 - settings.eps))
            - self._shift[group_name][feature_idx]
        )

    @Likelihood._api
    def get_dispersion(self, moment: Literal["mean", "std"] = "mean") -> pd.Series:
        """Get the dispersion vectors for each view.

        Args:
            moment: Which moment of the posterior distribution to return.
        """
        return pd.Series(getattr(self._dispersion, moment), index=self._feature_names)
