from typing import Literal, TypeAlias

from .base import API, APIType, Prior
from .gaussian_process import GaussianProcess, SmoothOptions
from .horseshoe import InformedHorseshoe
from .spike_slab import SpikeSlab

FactorPriorType: TypeAlias = Literal[*Prior.known_factor_priors()]
WeightPriorType: TypeAlias = Literal[*Prior.known_weight_priors()]
