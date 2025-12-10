import logging

from . import pl, tl
from ._core import MOFAFLEX, DataOptions, FeatureSet, FeatureSets, ModelOptions, TrainingOptions, settings
from ._core.api import priors
from ._version import __version__, __version_tuple__

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

if not logging.root.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s\t%(message)s"))
    _logger.addHandler(_handler)
    _logger.propagate = False  # https://github.com/pyro-ppl/pyro/pull/3422
