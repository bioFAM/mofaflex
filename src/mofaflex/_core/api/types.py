from ..terms import TermWrapper
from ..utils import building_docs
from . import terms

if building_docs():
    for term in dir(terms):
        globals()[term] = getattr(terms, term)
else:
    for term in dir(terms):
        globals()[term] = TermWrapper
