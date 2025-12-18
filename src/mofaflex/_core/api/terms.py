from ..terms import Term
from ._generate import init_api

init_api(__name__, Term, Term.known_terms)
