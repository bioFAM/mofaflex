from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mofaflex import MOFAFLEX
    from .base import Term


class TermWrapper:
    def __init__(self, model: MOFAFLEX, term: Term):
        self._model = model
        self._term = term

    def __dir__(self):
        return chain(self._model.__dir__(), self._term.api())

    def __getattr__(self, name):
        if name in self._term.api():
            return getattr(self._term, name)
        else:
            try:
                return getattr(self._model, name)
            except AttributeError as e:
                raise AttributeError(
                    f"'{self._term.__class__.__name__}' object has no attribute '{name}'", name=name, obj=self._term
                ) from e
