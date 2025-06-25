from typing import Literal, TypeAlias

from .base import Prior
from .gaussian_process import GP
from .horseshoe import Horseshoe
from .simple_location_scale import *  # noqa F403
from .spike_slab import SnS

FactorPriorType: TypeAlias = Literal[*Prior.known_factor_priors()]
WeightPriorType: TypeAlias = Literal[*Prior.known_weight_priors()]
