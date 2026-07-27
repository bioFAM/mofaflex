(modeldescription)=
# The MOFA-FLEX model

MOFA-FLEX aims to be a general framework for factor analysis of multi-omics data.
It contains features of MOFA{cite:p}`pmid29925568,pmid32393329`, MEFISTO{cite:p}`pmid35027765`, MuVI{cite:p}`pmlr-v206-qoku23a`, nonnegative matrix factorization, and NSF{cite:p}`pmid36587187`.
It allows users to flexibly combine features from all aforementioned methods.
MOFA-FLEX is a probabilistic Bayesian model and uses approximate variational inference{cite:p}`1601.00670v9,JMLR:v18:16-107,pmlr-v33-ranganath14,1301.1299v1,1610.05735v1,JMLR:v20:18-403` to flexibly handle large datasets and specifics of different data modalities.

In MOFA-FLEX we follow MOFA's terminology and refer to data modalities (groups of distinct features) as views and to batches of observations as groups.
Given a view $m$ and a group $g$ with $N_g$ observations and $D_m$ features and a user-defined parameter $K \in \mathbb{N}$, we decompose the observation matrix $\mat{Y}^{mg} \in \mathbb{R}^{N_g \times D_m}$ into a product of a factor matrix $\mat{Z}^{(g)} \in \mathbb{R}^{N_g \times K}$ and a weight matrix $\mat{W}^{(m)} \in \mathbb{R}^{K \times D_m}$, such that $\mat{Y}^{(m,g)} \approx \mat{Z}^{(g)} \mat{W}^{(m)}$.

## Choosing priors

As a Bayesian model, MOFA-FLEX has prior and posterior distributions for each model parameter.
The posterior family can currently not be influenced by the user, and parameters of the posterior are optimized during model training.
However, the prior can be selected by the user, and the choice of prior distribution can drastically affect the results.
MOFA-FLEX offers several prior options for both weights and factors, where each entry of the respective matrix is sampled independently:

- Normal: The standard Normal distribution $\Normal{0}{1}$.

- Laplace: The Laplace distribution $\Laplace{0}{1}$.

- Horseshoe: A variant of the Horseshoe distributionh{cite:p}`pmlr-v5-carvalho09a,10.2307/25734098`.
  This is a continuous distribution that has both infinite density at 0 and heavy tails.
  The Horseshoe prior thus exhibits strong shrinkage to 0 for small values, but simultaneously allows large values to escape shrinkage almost completely.
  It is thus a sparsity inducing prior and suitable if one operates under the assumption that a factor is active in only a few observations (when used as a factor prior) or features (when used as a weight prior).
  The density of the Horseshoe has no closed-form expression, and the distribution is defined hierarchically:
  \begin{align*}
  \lambda_i &\sim \HalfCauchy{0}{1}\\
  a_i &\sim \Normal{0}{\lambda_i^2\tau^2}
  \end{align*}
  where $\tau$ is a global shrinkage parameter, $\lambda_i$ are local shrinkage parameters, and $a_i$ are samples from the Horseshoe, in our case the entries of $\mat{W}$ or $\mat{Z}$.
  In practice, $\tau$ is often drawn from a half-Cauchy distribution: $\tau \sim \HalfCauchy{0}{1}$.
  Intuitively, the global shrinkage parameter $\tau$ pushes the entire distribution towards 0, while large values of individual local shrinkage parameters $\lambda_i$ can allow the corresponding samples to escape shrinkage{cite:p}`10.1093/acprof:oso/9780199694587.003.0017`.

  MOFA-FLEX implements the group Horseshoe{cite:p}`1709.04333v3` to allow shrinkage for entire views or groups of observations.
  Additionally, MOFA-FLEX's Horseshoe prior incorporates regularization{cite:p}`10.1214/17-EJS1337SI` to prevent individual weights from getting too large.
  In particular, our formulation of the regularized group Horseshoe is defined as
  \begin{align*}
  \tau_i &\sim \HalfCauchy{0}{1}\\
  \zeta_{i,k} &\sim \HalfCauchy{0}{1}\\
  \lambda_{i,k,j} &\sim \HalfCauchy{0}{1}\\
  c_{i,k,j} &\sim \InvGamma{0.5}{0.5}\\
  \beta_{i,k,j} &= \tau_i^2 \zeta_{i,k}^2 \lambda_{i,k,j}^2\\
  \sigma_{i,k,j}^2 &= \frac{c_{i,k,j} \beta_{i,k,j}}{c_{i,k,j} + \beta_{i,k,j}}\\
  a_{i,k,j} &\sim \Normal{0}{\sigma_{i,k,j}^2}
  \end{align*}
  where $i$ indexes the views or groups for weights and factors, respectively, $k$ indexes the factors, $j$ indexes the features or observations for weights and factors, respectively, and $a_{i,k,j}$ is $W^{(i)}_{k,j}$ or $Z^{(i)}_{j,k}$ for weights and factors, respectively.
  The intuition here is that if $\beta_{i,k,j} \ll c_{i,k,j}$, then $\sigma_{i,k,j}^2 \approx \beta_{i,k,j}$ and the prior approximates the original Horseshoe.
  However, if $\beta_{i,k,j} \gg c_{i,k,j}$, then $\sigma_{i,k,j}^2 \approx c_{i,k,j}$, thus regularizing very large coefficients as a Gaussian with variance $c_{i,k,j}$.
  This is the same prior as used by MuVI{cite:p}`pmlr-v206-qoku23a`.

- Spike and slab: A discrete sparsity-inducing prior with automatic relevance determination (ARD) for the non-zero slab component.
  This is a mixture of a Dirac delta distribution at 0 and a Normal distribution at 0 with positive variance:
  \begin{align*}
  \psi_{i,k} &\sim \dGamma{10^{-3}}{10^{-3}}\\
  \theta_{i,k} &\sim \dBeta{1}{1}\\
  p_{i,k,j} &\sim \dBernoulli{\theta_{i,k}}\\
  \beta_{i,k,j} &\sim \Normal{0}{\flatfrac{1}{\psi_{i,k}^2}}\\
  a_{i,k,j} &= p_{i,k,j}\beta_{i,k,j}
  \end{align*}
  where again $i$ indexes the views or groups for weights and factors, respectively, $k$ indexes the factors, $j$ indexes the features or observations for weights and factors, respectively, and $a_{i,k,j}$ is $W^{(i)}_{k,j}$ or $Z^{(i)}_{j,k}$ for weights and factors, respectively.
  To enable automatic differentiation variational inference, the variation posterior of $p_{i,k,j}$ is a ReinMax distribution{cite:p}`2304.08612v1`: An extension of the Bernoulli distribution with a second-order gradient approximation.
  The spike component $p$ induces sparsity for each individual value $a_{i,k,j}$, while the ARD component $\psi_{i,k}$ induces factor-wise sparsity.
  This is the same prior as used by MOFA{cite:p}`pmid29925568,pmid32393329`.

  In our experience, the ReinMax distribution typically requires higher learning rates or longer training durations than other, continuous, distributions.

- Gaussian process: A smoothness-inducing prior requiring additional covariates such as time or spatial coordinates for each observation.
  MOFA-FLEX considers two smoothness levels: Between observations and between groups.
  This is achieved by using a product kernel of kernels for within-group and between-group smoothness.
  In detail, the group covariance matrix for $G$ groups and factor $k$ is defined by a low-rank approximation
  \begin{equation*}
  \widetilde{\mat{K}}^{(k)} = \sum_{r=1}^R \vec{x}_r^{(k)} {\vec{x}_r^{(k)}}\trns + \sigma_k^2 \eye_G \qq{with $\vec{x}_r^{(k)} \in \mathbb{R}^G$}
  \end{equation*}
  where $\vec{x}_r^{(k)}$ and $\sigma_k^2$ are free parameters learned during optimization.
  By normalizing $\widetilde{\mat{K}}^{(k)}$ we obtain the group correlation matrix $\mat{K}^{(k)}$.

  Given a kernel function $\widetilde{\kappa}(\vec{c}_n^g, \vec{c}_{n'}^{g'})$, we can define the complete GP kernel for factor $k$ as
  \begin{equation*}
  \kappa_k(\vec{c}_n^g, \vec{c}_{n'}^{g'}) = (1 - \zeta_k) \widetilde{\kappa}(\vec{c}_n^g, \vec{c}_{n'}^{g'}) K^{(k)}_{g, g'}
  \end{equation*}
  The factor values are then given by
  \begin{align*}
  \eta_{g, n, k} &\sim \Normal{0}{\zeta_k}\\
  f_k &\sim \GP(0, \kappa_k)\\
  Z^{(g)}_{n, k} &= f_k(\vec{c}_n^g) + \eta_{g,n,k}
  \end{align*}
  $\zeta_k \in (0, 1)$ is a free parameter learned during optimization and determines the degree of smoothness: For $\zeta_k \to 0$, all variance of $\vec{Z}^{(g)}_{:, k}$ is explained by the Gaussian process, while for $\zeta_k \to 0$, all variance is explained by random noise.

  MOFA-FLEX offers a choice between two kernel functions $\widetilde{\kappa}$: The squared exponential or RBF kernel $\widetilde{\kappa}_k(\vec{c}_n^g, \vec{c}_{n'}^{g'}) = \exp(-\frac{1}{2}d_k)$ and the Matérn kernel $\widetilde{\kappa_k}(\vec{c}_n^g, \vec{c}_{n'}^{g'}) = \frac{2^{1-\nu}}{\Gamma(\nu)} \left( \sqrt{2 \nu}d_k \right)^{\nu} K_{\nu} \left( \sqrt{2 \nu}d_k \right)$ where $K_\nu$ is the modified Bessel function and $\nu \in \{0.5, 1.5, 2.5\}$ is a user-defined hyperparameter. $d = \frac{\norm{\vec{c}_n^g - \vec{c}_{n'}^{g'}}_2^2}{l_k^2}$ is the scaled distance, with $l_k$ a free parameter estimated during optimization.
  MOFA-FLEX with the RBF kernel corresponds to the formulation of MEFISTO{cite:p}`pmid35027765` and with the Matérn kernel to nonnegative spatial factorization{cite:p}`pmid36587187`.

  For 1-dimensional covariates, which often represent time, MOFA-FLEX supports dynamic time warping to align multiple potentially mismatched timeseries.
  The algorithm and implementation closely follow MEFISTO{cite:p}`pmid35027765`.
  However, MOFA-FLEX's time warping algorithm works for any GP kernel.

- Guided sparse factor analysis (GSFA, for factors only): A prior custom-made for CRISPR perturbation screens.
  See {cite:t}`pmid37770710` for details.
  Briefly, this prior requires an assignment of observations to perturbations, given by the perturbation matrix $\mat{G} \in \mathbb{R}^{N_g \times T}$ where $T$ is the number of targets.
  The factors are then given by $\mat{G}\mat{\beta}$ where $\mat{\beta} \in \mathbb{R}^{T \times K}$ represents the effects of perturbations on factors.
  Each row of $\mat{\beta}$ is sampled from a spike and slab distribution similar to the one described above.

## Choosing likelihoods

The likelihood is the probability distribution of the observed data $\mat{Y}^{(m,g)}$, parametrized by the mean $\mat{Z}^{(g)} \mat{W}^{(m)}$ and, depending on the likelihood, a link function transforming the real-valued matrix product into nonnegative or bounded values and additional parameters sampled from their own prior distributions.

MOFA-FLEX offers a choice of three likelihoods:

- Normal: The Normal distribution $\Normal{\mu}{\sigma^2}$. This should be suitable for most types of real-valued data.

- negative binomial: The negative binomial distribution $\NegativeBinomial{\mu}{\gamma}$ with mean $\mu$ and overdispersion $\gamma$.
  This should be suitable for count data such as sequencing.

- Bernoulli: The Bernoulli distribution modeling binary data. This should be suitable for metylation or ATAC data.

If no likelihood is specified by the user, MOFA-FLEX tries to automatically determine a suitable likelihood based on the data using a simple heuristic:
If all values in a view are 0 or 1, the Bernoulli likelihood is chosen.
Otherwise, if all values are integers, the Gamma-Poisson likelihood is selected.
The Normal likelihood is used in all other cases.

## Nonnegative factors

Factors and can be forced to be nonnegative, which can lead to more interpretable results{cite:p}`pmid36587187`.
This is achieved by applying a ReLU function to the real-valued factor or weight values.

## Using prior domain knowledge

MOFA-FLEX fully supports the approach pioneered in MuVI{cite:p}`pmlr-v206-qoku23a` to use prior domain knowledge.
Given one or multiple feature sets, consisting of for example genes belonging to the same biochemical pathways, the goal is to reserve one factor for each feature set, forcing weights for all features not included in the respective gene set to 0.
However, domain knowledge is often imperfect, thus the feature sets may be incomplete or contain superfluous features.
The model therefore allows a measure of noisiness: Weights for features not included in a feature set are strongly shrunk towards zero, but are allowed to be non-zero if strongly supported by the data.
Similarly, weights for features included in a gene set can approach zero.

In practice, this is achieved by slightly modifying the Horseshoe prior:
Given a feature set $S_k$ and a user-defined parameter $\widetilde\alpha \in (0, 1)$, we set
\begin{align*}
\alpha_{k,j} &= \begin{cases}1 & j \in S_k\\\widetilde{\alpha} & j \notin S_k\end{cases}\\
\widetilde{c}_{i,k,j} &= \alpha_{k,j} c_{i,k,j}
\end{align*}
for view $i$, factor $k$, and feature $j$.
We then redefine $\sigma^2$ as
\begin{equation*}
\sigma^2_{i,j,k} = \frac{\widetilde{c}_{i,k,j} \beta_{i,k,j}}{\widetilde{c}_{i,k,j} + \beta_{i,k,j}}
\end{equation*}

As should be obvious from the above, domain knowledge is only supported when using the Horseshoe prior for the weights.

## Guiding individual factors using external covariates

SOFA{cite:p}`capraz2024semi` pioneered the coupling of individual factors to external covariates in multi-view factor models, thereby facilitating the disentanglement of known and unknown sources of variation. This coupling is achieved using an additional linear regression task from the external covariate, which can be assumed to follow a Normal, Bernoulli or Categorical distribution, on the factor. Several covariates can be provided and each is assigned to a unique factor.

Given the factor scores $Z_{n,k}$ of a single factor $k$ and observation $n$, the linear predictor $\nu_n$ for a given covariate is given by
\begin{align*}
    \nu_n = w_0 + w_1 Z_{n,k},
\end{align*}
where $w_0$ is an intercept and $w_1$ is a regression coefficient. $w_0$ and $w_1$ are Bayesian latent variables with standard Normal prior distributions:
\begin{align*}
    w_i \sim \Normal{0}{1} \quad i \in [0, 1].
\end{align*}
The covariate is then modeled using Normal, Bernoulli or Categorical distributions with the linear predictor $\nu_n$ (and an additional scale parameter in the Normal distribution.)
