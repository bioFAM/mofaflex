# Configuration file for the Sphinx documentation builder.

# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
import sys
import os
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path

os.environ["MOFAFLEX_DOCS"] = "1"

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "extensions"))


# -- Project information -----------------------------------------------------

# NOTE: If you installed your project in editable mode, this might be stale.
#       If this is the case, reinstall it to refresh the metadata
info = metadata("mofaflex")
project = info["Name"]
author = info["Author-email"]
copyright = f"{datetime.now():%Y}, {author}."
version = info["Version"]
urls = dict(pu.split(", ") for pu in info.get_all("Project-URL"))
repository_url = urls["Source"]

# The full version, including alpha/beta/rc tags
release = info["Version"]

bibtex_bibfiles = ["references.bib"]
templates_path = ["_templates"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.0"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "bioFAM",
    "github_repo": project,
    "github_version": "main",
    "conf_py_path": "/docs/",
}

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings.
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinx_autodoc_typehints",
    "sphinx_tabs.tabs",
    "sphinx.ext.mathjax",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinxext.opengraph",
    *[p.stem for p in (HERE / "extensions").glob("*.py")],
]

autosummary_generate = True
autosummary_imported_members = True
autodoc_member_order = "groupwise"
default_role = "literal"
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
myst_heading_anchors = 6  # create anchors for h1-h6
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "amsmath", "html_image", "html_admonition"]
myst_url_schemes = ("http", "https", "mailto")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {".rst": "restructuredtext", ".ipynb": "myst-nb", ".myst": "myst-nb"}

intersphinx_mapping = {
    # TODO: replace `3.13` with `3` once ReadTheDocs supports building with Python 3.14
    "python": ("https://docs.python.org/3.13", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "mudata": ("https://mudata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "plotnine": ("https://plotnine.org/", None),
    "pytorch": ("https://docs.pytorch.org/docs/stable/", None),
    "muon-tutorials": ("https://muon-tutorials.readthedocs.io/en/latest", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

html_title = project

html_theme_options = {
    "repository_url": repository_url,
    "use_repository_button": True,
    "path_to_docs": "docs/",
    "navigation_with_keys": False,
}

pygments_style = "default"

nitpick_ignore = [
    # If building the documentation fails because of a missing link that is outside your control,
    # you can add an exception to this list.
    #     ("py:class", "igraph.Graph"),
]

ogp_site_url = "https://mofaflex.readthedocs.io/stable/"
ogp_image = "_static/img/mofaflex_schematic.png"


# -- MathJax macros -----------------------------------------------------------
mathjax3_config = {
    "tex": {
        "macros": {
            "vec": ["\\boldsymbol{\\mathrm{#1}}", 1],
            "mat": ["\\boldsymbol{\\mathrm{#1}}", 1],
            "zeros": "\\vec{0}",
            "ones": "\\vec{1}",
            "eye": "\\mat{I}",
            "trns": "^\\mathrm{T}",
            "itrns": "^{-\\mathrm{T}}",
            "cond": "\\;\\middle|\\;",
            "Uniform": ["\\mathcal{U}\\left(#1, #2\\right)", 2],
            "Normal": ["\\mathcal{N}\\left(#1, #2\\right)", 2],
            "Lognormal": ["\\mathrm{Lognormal}\\left(#1, #2\\right)", 2],
            "Laplace": ["\\mathrm{Laplace}\\left(#1, #2\\right)", 2],
            "dGamma": ["\\mathcal{G}\\left(#1, #2\\right)", 2],
            "InvGamma": ["\\mathrm{Inv}\\mathcal{G}\\left(#1, #2\\right)", 2],
            "dBeta": ["\\mathrm{Beta}\\left(#1, #2\\right)", 2],
            "Cauchy": ["C\\left(#1, #2\\right)", 2],
            "HalfCauchy": ["C^+\\left(#1, #2\\right)", 2],
            "dMultinomial": ["\\mathrm{Multinomial}\\left(#1\\right)", 1],
            "Dirichlet": ["\\mathrm{Dir}\\left(#1\\right)", 1],
            "Categorical": ["\\mathrm{Cat}\\left(#1\\right)", 1],
            "dExp": ["\\mathrm{Exp}\\left(#1\\right)", 1],
            "dBernoulli": ["\\mathrm{Ber}\\left(#1\\right)", 1],
            "Poisson": ["\\mathrm{Pois}\\left(#1\\right)", 1],
            "NegativeBinomial": ["\\mathrm{NB}\\left(#1, #2\\right)", 2],
            "GammaPoisson": ["\\mathrm{GamPoi}\\left(#1, #2\\right)", 2],
            "HorseShoe": ["\\mathrm{HS}\\left(#1\\right)", 1],
            "HorseShoePlus": ["\\mathrm{HS+}\\left(#1\\right)", 1],
            "dDiracDelta": ["\\delta_{#1}", 1],
            "KL": ["\\mathrm{KL}\\left(#1\\parallel#2\\right)", 2],
            "Exp": "\\operatorname{\\mathbb{E}}",
            "Var": "\\operatorname{\\mathbb{V}ar}",
            "Cov": "\\operatorname{\\mathbb{C}ov}",
            "Prob": "\\operatorname{\\mathbb{P}}",
            "diag": "\\operatorname{diag}",
            "GP": "\\operatorname{\\mathcal{GP}}",
            "relu": "\\operatorname{ReLU}",
        },
        "packages": {"[+]": ["physics", "mathtools"]},
    },
    "loader": {"load": ["[tex]/physics", "[tex]/mathtools"]},
}
