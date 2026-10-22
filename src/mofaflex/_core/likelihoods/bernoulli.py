import logging
from collections.abc import Mapping
from types import ModuleType

import numpy as np
from anndata import AnnData
from scipy.special import expit, logit

from ..datasets import MofaFlexDataset
from ..settings import settings
from ..utils import Matrix, Vector, nanmean
from .base import R2, Likelihood
from .pyro import Bernoulli as PyroBernoulli
from .pyro import Likelihood as PyroLikelihood

_logger = logging.getLogger(__name__)


class Bernoulli(Likelihood):
    """Bernoulli likelihood for binary data."""

    _priority = 10
    _state_attrs = ("_shift",)

    @staticmethod
    def _calc_shift(adata: AnnData):
        shift = logit(nanmean(adata.layers[None], axis=0))
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

        if nonnegative:
            _logger.warning(
                f"The {self.__class__.__name__} likelihood does not support nonnegative views. Your results may be suboptimal."
            )

    def _get_pyro_likelihood(self, data: MofaFlexDataset, sample_dim: int, feature_dim: int) -> PyroLikelihood:
        return PyroBernoulli(
            self._view_name,
            sample_dim,
            feature_dim,
            data.n_samples,
            data.n_features[self._view_name],
            shift=self._shift,
        )

    def on_train_end(self, *args, **kwargs):
        self._shift = self._impute_feature_summary_statistics(self._shift)

    @classmethod
    def _validate(cls, data: Mapping[str | None, tuple[Vector[np.number]], ModuleType], view_name: str) -> str | None:
        if data.keys() != {None}:
            return f"The {cls.__name__} likelihood requires one layer, got {len(data)} for view {view_name}."
        data, xp = data[None]
        if not xp.all(xp.isclose(data, 0) | xp.isclose(data, 1)):  # TODO: set correct atol value
            return f"The {cls.__name__} likelihood in view {view_name} must be used with binary data."
        return None

    def _r2_impl(
        self,
        y_true: Mapping[str | None, Matrix[np.number]],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> R2:
        y_true = y_true[None]
        y_pred = self.transform_prediction(y_pred, group_name, sample_idx, feature_idx)
        ss_res = np.nansum(self._dV_square(y_true, y_pred, -1, 1))
        ss_tot = np.nansum(self._dV_square(y_true, expit(self._shift[group_name][feature_idx]), -1, 1))
        return R2(ss_res, ss_tot)

    @staticmethod
    def _logpmf(y, logits):
        m1 = np.maximum(0, -logits)
        m2 = np.maximum(0, logits)
        return -y * (m1 + np.log(np.exp(0 - m1) + np.exp(-logits - m1))) - (1 - y) * (
            m2 + np.log(np.exp(0 - m2) + np.exp(logits - m2))
        )

    def _deviance_explained_impl(
        self,
        y_true: Mapping[str | None, Matrix[np.number]],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> R2:
        y_true = y_true[None]
        # the shift is part of the linear predictor, see transform_prediction
        shift = self._shift[group_name][feature_idx]
        loglik_null = np.nansum(self._logpmf(y_true, shift))
        loglik_model = np.nansum(self._logpmf(y_true, y_pred + shift))

        # saturated model has deviance 0
        return R2(loglik_model, loglik_null)

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
        return logit(np.clip(data[None], settings.eps, 1 - settings.eps)) - self._shift[group_name][feature_idx]
