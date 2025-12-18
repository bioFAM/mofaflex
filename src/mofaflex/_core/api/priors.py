from ..priors import Prior
from ._generate import init_api

init_api(__name__, Prior, Prior.known_priors())
