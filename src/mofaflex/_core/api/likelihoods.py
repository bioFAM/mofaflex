from ..likelihoods import Likelihood
from ._generate import init_api

init_api(__name__, Likelihood, Likelihood.known_likelihoods())
