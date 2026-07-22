from typing import Literal, TypeAlias

from .base import Prior, PriorDynamicAPI
from .constant import Constant
from .gaussian_process import GaussianProcess
from .gsfa import GSFA
from .horseshoe import Horseshoe, InformedHorseshoe
from .simple_location_scale import *  # noqa F403
from .simplex import Simplex
from .spike_slab import SpikeSlab

FactorPriorType: TypeAlias = Literal[*Prior.known_priors("factors")]
WeightPriorType: TypeAlias = Literal[*Prior.known_priors("weights")]
