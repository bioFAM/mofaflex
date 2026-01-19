# MOFA-FLEX

[![PyPI][badge-pypi]][pypi]
[![Tests][badge-tests]][tests]
[![codecov][badge-codecov]][codecov]
[![Documentation][badge-docs]][documentation]

[badge-pypi]: https://img.shields.io/pypi/v/mofaflex
[badge-tests]: https://github.com/bioFAM/mofaflex/actions/workflows/test.yaml/badge.svg
[badge-codecov]: https://codecov.io/gh/bioFAM/mofaflex/graph/badge.svg?token=IJP1IA4JEU
[badge-docs]: https://img.shields.io/readthedocs/mofaflex


![graphical abstract](https://raw.githubusercontent.com/bioFAM/mofaflex/main/docs/_static/img/mofaflex_schematic.svg)

MOFA-FLEX is a versatile factor analysis framework designed to streamline the construction and training of complex matrix factorisation models for omics data.
It is built on a probabilistic programming-based Bayesian factor analysis framework that integrates concepts from multiple existing methods while remaining modular and extensible.
MOFA-FLEX generalises widely used matrix factorisation tools by incorporating flexible prior options (including structured sparsity priors for multi-omics data and covariate-informed priors for spatio-temporal data), non-negativity constraints, and diverse data likelihoods - allowing users to mix and match components to suit their specific needs.
Additionally, MOFA-FLEX introduces a novel module for integrating prior biological knowledge in the form of gene sets or, more generally, variable sets, enabling the inference of interpretable latent factors linked to specific molecular programs.

## Getting started

Please refer to the [documentation][]. In particular, the

- [Getting started guide][getting started].
- [Tutorials][].
- [API documentation][].

## Installation

You need to have Python 3.11 or newer installed on your system. If you don't have
Python installed, we recommend installing [uv][].

There are several alternative options to install MOFA-FLEX:

1. Install the latest release of MOFA-FLEX from [PyPI][]:

```bash
pip install mofaflex
```

2. Install the latest development version:

```bash
pip install git+https://github.com/bioFAM/mofaflex.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [discussions][].
If you found a bug, please use the [issue tracker][].

## Citation

If you use MOFA-FLEX in your work, please cite
> Qoku A, Rohbeck M, Walter FC, Kats I, Stegle O, and Buettner F.  MOFA-FLEX: A Factor Model Framework for Integrating Omics Data with Prior Knowledge. Preprint at [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.11.03.686250) (2025). DOI: [10.1101/2025.11.03.686250](https://doi.org/10.1101/2025.11.03.686250).

<details><summary>BibTeX</summary>

```bibtex
@article {mofaflex,
	author = {Qoku, Arber and Rohbeck, Martin and Walter, Florin Cornelius and Kats, Ilia and Stegle, Oliver and Buettner, Florian},
	title = {MOFA-FLEX: A Factor Model Framework for Integrating Omics Data with Prior Knowledge},
	eprint = {2025.11.03.686250},
	year = {2025},
	doi = {10.1101/2025.11.03.686250},
	URL = {https://www.biorxiv.org/content/early/2025/11/04/2025.11.03.686250},
	archiveprefix = {bioRxiv}
}
```
</details>

[uv]: https://github.com/astral-sh/uv
[issue tracker]: https://github.com/bioFAM/mofaflex/issues
[tests]: https://github.com/bioFAM/mofaflex/actions/workflows/test.yaml
[codecov]: https://codecov.io/gh/bioFAM/mofaflex
[documentation]: https://mofaflex.readthedocs.io
[discussions]: https://github.com/bioFAM/mofaflex/discussions
[changelog]: https://mofaflex.readthedocs.io/stable/changelog.html
[getting started]: https://mofaflex.readthedocs.io/stable/notebooks/getting_started.html
[tutorials]: https://mofaflex.readthedocs.io/stable/tutorials.html
[api documentation]: https://mofaflex.readthedocs.io/stable/api/index.html
[pypi]: https://pypi.org/project/mofaflex
