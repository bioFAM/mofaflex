from typing import Literal, TypeAlias

from .base import API, APIType, Prior
from .gaussian_process import GaussianProcess
from .horseshoe import InformedHorseshoe
from .simple_location_scale import *  # noqa F403
from .spike_slab import SpikeSlab

FactorPriorType: TypeAlias = Literal[*Prior.known_priors("factors")]
WeightPriorType: TypeAlias = Literal[*Prior.known_priors("weights")]
