from collections.abc import Iterable
from typing import Literal

import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import IndexKernel, Kernel, MaternKernel, RBFKernel, ScaleKernel
from gpytorch.means import ZeroMean
from gpytorch.models import ApproximateGP
from gpytorch.priors import Prior
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from numpy.typing import NDArray


class BasicKernel(Kernel):
    """A kernel that does not model group correlations.

    This kernel should be used when covariates are not aligned across groups.
    """

    def __init__(self, base_kernel: Kernel | None, n_groups: int):
        super().__init__()

        self.base_kernel = base_kernel
        self.n_groups = n_groups

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, last_dim_is_batch=False, **params):
        x1_, x2_ = x1[..., 1:], x2[..., 1:]
        return self.base_kernel(x1_, x2_, diag, last_dim_is_batch, **params)

    # diagonal group covariance matrix for compatibility with MefistoKernel
    @property
    def group_corr(self):
        return torch.eye(self.n_groups)[None, ...].expand(self.base_kernel.batch_shape[0], -1, -1)


class MefistoKernel(Kernel):
    """A kernel that combines a base kernel with group-specific correlations.

    This kernel implements a combination of a base kernel with
    a learned group correlation structure through an IndexKernel.

    Args:
        base_kernel: The base kernel to use for computing similarities between inputs.
        n_groups: Number of groups to model correlations between.
        rank: Rank of the low-rank approximation for group correlations.
        lowrank_covar_prior: Optional prior for the low-rank covariance.
        **kwargs: Additional arguments passed to parent class.
    """

    def __init__(
        self,
        base_kernel: Kernel | None,
        n_groups: int,
        rank: int = 1,
        lowrank_covar_prior: Prior | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.group_kernel = IndexKernel(
            num_tasks=n_groups, batch_shape=base_kernel.batch_shape, rank=rank, prior=lowrank_covar_prior
        )
        self.base_kernel = base_kernel

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag=False, last_dim_is_batch=False, **params):
        group_idx1, group_idx2 = x1[..., 0, None], x2[..., 0, None]
        x1_, x2_ = x1[..., 1:], x2[..., 1:]
        base_mat = self.base_kernel(x1_, x2_, diag, last_dim_is_batch, **params)

        if not diag:
            group_cov = self.group_kernel(group_idx1, group_idx2)
            if x1 is x2 or x1.shape == x2.shape and x1.data_ptr() == x2.data_ptr():
                group_cov_diag1 = group_cov_diag2 = group_cov.diagonal().sqrt()
            else:
                group_cov_diag1 = self.group_kernel(group_idx1, diag=True).sqrt()
                group_cov_diag2 = self.group_kernel(group_idx2, diag=True).sqrt()
            group_cor = group_cov.div(group_cov_diag1[..., None]).div(group_cov_diag2[..., None, :])
            return base_mat.mul(group_cor)
        else:
            return base_mat

    @property
    def group_corr(self):
        covar = self.group_kernel.covar_matrix.to_dense()
        diag = covar.diagonal(dim1=-1, dim2=-2).sqrt()
        return covar / diag[..., None] / diag[..., None, :]


class GP(ApproximateGP):
    """Gaussian Process model.

    A variational Gaussian Process model that combines a base kernel with group-specific
    correlations through a MefistoKernel.

    Args:
        n_inducing: Number of inducing points.
        covariates: Iterable of covariate tensors to choose inducing points from.
        n_factors: Number of factors in the model.
        n_groups: Number of groups to model correlations between.
        kernel: Kernel type.
    i   independent_lengthscales: Whether to use a separate lengthscale for each covariate dimension.
        rank: Rank of the group correlation kernel.
        **kwargs: Additional kernel-specific parameters.
    """

    def __init__(
        self,
        n_inducing: int,
        covariates: Iterable[NDArray],
        n_factors: int,
        n_groups: int,
        kernel: Literal["RBF", "Matern"] = "RBF",
        independent_lengthscales: bool = False,
        rank: int = 1,
        use_mefisto_kernel: bool = True,
        **kwargs,
    ):
        covariates = tuple(covariates)
        self._inducing_points_idx = get_inducing_points_idx(covariates, n_inducing, n_factors)
        self._n_inducing = n_inducing

        inducing_points = setup_inducing_points(covariates, self._inducing_points_idx, n_inducing)
        if inducing_points.shape[-3] != n_factors:
            raise ValueError("The first dimension of inducing_points must be n_factors.")

        n_dims = inducing_points.shape[-1]

        variational_distribution = CholeskyVariationalDistribution(n_inducing, batch_shape=(n_factors,))

        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=False
        )

        super().__init__(variational_strategy)

        self.mean_module = ZeroMean(batch_shape=(n_factors,))

        max_dist = torch.pdist(inducing_points.flatten(0, 1), p=n_dims).max()

        kernel_kwargs = {"batch_shape": (n_factors,), "lengthscale_constraint": Interval(max_dist / 20, max_dist)}
        init_shape = (n_factors,)
        if independent_lengthscales:
            kernel_kwargs["ard_num_dims"] = n_dims
            init_shape = (n_factors, n_dims)

        if kernel == "RBF":
            base_kernel = RBFKernel(**kernel_kwargs)
        elif kernel == "Matern":
            base_kernel = MaternKernel(nu=kwargs.get("nu", 1.5), **kernel_kwargs)
        base_kernel.lengthscale = max_dist * torch.rand(*init_shape).clamp(0.1)

        base_kernel = ScaleKernel(
            base_kernel, outputscale_constraint=Interval(1e-3, 1 - 1e-3), batch_shape=(n_factors,)
        )
        base_kernel.outputscale = torch.sigmoid(torch.randn(n_factors)).clamp(1e-3, 1 - 1e-3)

        if use_mefisto_kernel:
            self._covar_module = MefistoKernel(base_kernel, n_groups, rank)
        else:
            self._covar_module = BasicKernel(base_kernel, n_groups)

    @property
    def outputscale(self):
        return self._covar_module.base_kernel.outputscale

    @property
    def lengthscale(self):
        return self._covar_module.base_kernel.base_kernel.lengthscale.squeeze(-2)

    @property
    def group_corr(self):
        return self._covar_module.group_corr

    def __call__(self, input: tuple[torch.Tensor | None, torch.Tensor | None], prior: bool = False, **kwargs):
        group_idx, inputs = input
        if group_idx is not None and inputs is not None:
            inputs = torch.cat((group_idx, inputs), dim=-1)
        return super().__call__(inputs, prior, **kwargs)

    def forward(self, x):
        """Forward pass of the GP model."""
        mean = self.mean_module(x)
        covar = self._covar_module(x)
        return MultivariateNormal(mean, covar)

    def update_inducing_points(self, covariates):
        setup_inducing_points(
            covariates, self._inducing_points_idx, self._n_inducing, out=self.variational_strategy.inducing_points
        )


def get_inducing_points_idx(
    covariates: Iterable[torch.Tensor], n_inducing: int, n_factors: int
) -> tuple[tuple[torch.Tensor]]:
    """Generate random indices for selecting inducing points from covariates.

    Args:
        covariates: Iterable of covariate tensors.
        n_inducing: Number of inducing points to select.
        n_factors: Number of factors in the model.

    Returns:
        tuple: Nested tuple of indices for each factor and covariate.
    """
    n = [0] + [c.shape[0] for c in covariates]
    totaln = sum(n)
    offsets = torch.cumsum(torch.as_tensor(n), 0)
    idx = tuple(torch.randint(0, totaln, (n_inducing,)).sort().values for _ in range(n_factors))
    return tuple(
        tuple(cidx[(s <= cidx) & (cidx < e)] - s for s, e in zip(offsets[:-1], offsets[1:], strict=False))
        for cidx in idx
    )


def setup_inducing_points(covariates: Iterable[NDArray], idx, n_inducing, *, out=None) -> torch.Tensor:
    """Initialize inducing points from covariates using provided indices.

    Args:
        covariates: Iterable of covariate tensors.
        idx: Indices from get_inducing_points_idx.
        n_inducing: Number of inducing points.
        out: Optional pre-allocated tensor for output.

    Returns:
        torch.Tensor: Tensor of inducing points.
    """
    if covariates is None:
        return None

    covariates = tuple(torch.as_tensor(cov) for cov in covariates)
    group_idx = tuple(torch.as_tensor(i) for i in range(len(covariates)))

    if out is None:
        out = torch.empty((len(idx), n_inducing, 1 + covariates[0].shape[-1]))
    for factor, factoridx in enumerate(idx):
        offset = 0
        for cov, cidx, gidx in zip(covariates, factoridx, group_idx, strict=False):
            noffset = offset + cidx.shape[0]
            out[factor, offset:noffset, 1:] = cov[cidx]
            out[factor, offset:noffset, 0] = gidx
            offset = noffset
    return out
