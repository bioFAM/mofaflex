# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.1.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [0.1.1] (Unreleased)

### Added
- `tl.factor_correlation` to calculate the correlation between factors

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

- Initial release

[0.1.1]: https://github.com/bioFAM/mofaflex/releases/tag/v0.1.1
[0.1.0]: https://github.com/bioFAM/mofaflex/releases/tag/v0.1.0
