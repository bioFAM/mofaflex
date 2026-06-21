# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.2.0] (Unreleased)

### Added
- Support for multiple additive terms.
- Constant pseudo-prior. When used for weights, this can be used to project new data into an already existing latent space.
- GSFA prior for analysis of CRISPR perturbation screens.
- Single AnnData objects can now be used as input data. MOFA-FLEX will assume exactly one view for this type of input.
- MuData objects with `axis=1` can now be used as input data. MOFA-FLEX will treat each modality as a group and use the `group_by`
  argument, if given, to select a column in `.var` to split the data into views.
- `pl.variance_explained` gained an `annotated_only` argument to plot only the factors informed by prior annotations
  (e.g. the gene-set factors of an `InformedHorseshoe` prior), dropping the uninformed dense factors.

### Changed
- The `show_featurenames` argument to `pl.factor` is now called `show_samplenames` to better reflect what it actually does.
- The API has received an overhaul. Make sure to re-familiarize yourself with the tutorials.
- R2 estimation for non-Gaussian likelihoods should be more robust.
- The on-disk format for trained models has changed. Files created with mofaflex 0.1 cannot be read by 0.2 and vice versa.
- The Gaussian process prior can now also be used for weights.
- The spike and slab prior now has option to make the background distribution a Gaussian.
- Training with sparse inputs and minibatching is about 1.5 times faster.

### Fixed
- `pl.factor_significance` now ranks factors by the variance they explain within the selected `views`/`groups`,
  rather than the overall variance, so restricting `views` reorders the plot accordingly.
- `pl.top_weights` orders features per facet, fixing the scrambled ordering when a feature is among the top
  features of several factors at once.
- `MOFAFLEX.load` without an explicit `map_location` now falls back to the default device, so methods that need
  it (e.g. `pl.factor_significance`/PCGSE) work on reloaded models.

### Removed
- The MOFA compatibility mode for saving a trained model.

## [0.1.2] (Unreleased)

### Fixed
- Using a MuData `.obs` column as a guiding variable now works.
- When using guiding variables together with annotations, the data frame returned by `get_annotations` now has a correct row index.
- Compatibility with Pandas 3.

## [0.1.1]

### Added
- `tl.factor_correlation` to calculate the correlation between factors.

### Fixed
- Gaussian processes with dynamic time warping and a custom reference group now actually
  use the the set reference group instead of the first warped group as reference.
- The PCGSE test now also works if only a single annotated factor is present.

### Changed
- `pl.factor` now also accepts factor names for the `factor` argument.
- `FeatureSets.filter` now has better defaults (based on extensive benchmarking).

### Deprecated
- The MOFA compatibility mode for saving a trained model.

## [0.1.0]

- Initial release.

[0.2.0]: https://github.com/bioFAM/mofaflex/releases/tag/v0.2.0
[0.1.2]: https://github.com/bioFAM/mofaflex/releases/tag/v0.1.2
[0.1.1]: https://github.com/bioFAM/mofaflex/releases/tag/v0.1.1
[0.1.0]: https://github.com/bioFAM/mofaflex/releases/tag/v0.1.0
