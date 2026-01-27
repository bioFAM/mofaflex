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

### Changed
- The `show_featurenames` argument to `pl.factor` is now called `show_samplenames` to better reflect what it actually does.
- The API has received an overhaul. Make sure to re-familiarize yourself with the tutorials.
- R2 estimation for non-Gaussian likelihoods should be more robust.
- The on-disk format for trained model has changed. Files created with mofaflex 0.1 cannot be read by 0.2 and vice versa.

### Removed
- The MOFA compatibility mode for saving a trained model.

## [0.1.1] (Unreleased)

### Added
- `tl.factor_correlation` to calculate the correlation between factors.

### Fixed
- Gaussian processes with dynamic time warping and a custom reference group now actually
  use the the set reference group instead of the first warped group as reference.

### Changed
- `pl.factor` now also accepts factor names for the `factor` argument.

### Deprecated
- The MOFA compatibility mode for saving a trained model.

## [0.1.0]

- Initial release.

[0.2.0]: https://github.com/bioFAM/mofaflex/releases/tag/v0.2.0
[0.1.1]: https://github.com/bioFAM/mofaflex/releases/tag/v0.1.1
[0.1.0]: https://github.com/bioFAM/mofaflex/releases/tag/v0.1.0
