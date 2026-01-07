from typing import Literal, TypeAlias

from . import pyro
from .base import Likelihood
from .bernoulli import Bernoulli
from .negativebinomial import NegativeBinomial
from .normal import Normal

LikelihoodType: TypeAlias = Literal[*Likelihood.known_likelihoods().keys()]
