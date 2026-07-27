import numpy as np
from anndata import AnnData
from scipy.special import expit, logit

from ..datasets import MofaFlexDataset
from ..settings import settings
from ..utils import Matrix, Vector, nanmean
from .base import R2, Likelihood
from .pyro import Bernoulli as PyroBernoulli
from .pyro import Likelihood as PyroLikelihood


class Bernoulli(Likelihood):
    """Bernoulli likelihood for binary data."""

    _priority = 10
    _state_attrs = ("_shift",)

    @staticmethod
    def _calc_shift(adata: AnnData):
        shift = logit(nanmean(adata.X, axis=0))
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
    def _validate(cls, data: Matrix[np.number], xp) -> bool:
        return xp.all(xp.isclose(data, 0) | xp.isclose(data, 1))  # TODO: set correct atol value

    @classmethod
    def _format_validate_exception(cls, view_name: str) -> str:
        return f"Bernoulli likelihood in view {view_name} must be used with binary data."

    def _r2_impl(
        self,
        y_true: Matrix[np.number],
        y_pred: Matrix[np.floating],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> R2:
        ss_res = np.nansum(self._dV_square(y_true, y_pred, -1, 1))
        ss_tot = np.nansum(self._dV_square(y_true, expit(self._shift[group_name][feature_idx]), -1, 1))
        return R2(ss_res, ss_tot)

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
        data: Matrix[np.number],
        group_name: str,
        sample_idx: Vector[int] | slice = slice(None),
        feature_idx: Vector[int] | slice = slice(None),
    ) -> Matrix[np.floating]:
        return logit(np.clip(data, settings.eps, 1 - settings.eps)) - self._shift[group_name][feature_idx]
