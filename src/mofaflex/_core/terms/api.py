from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..mofaflex import MOFAFLEX
    from .base import Term


class TermWrapper:
    """Wrapper class for additive termsthat only exposes the user-facing API.

    If a requested attribute is not found in the term, the wrapper tries to get it from the main
    MOFAFLEX instance. This is helpful to be able to access things like `n_samples` and `n_features`
    directly from terms without also having access to the MOFAFLEX instance.
    """

    def __init__(self, model: MOFAFLEX, term: Term):
        self._model = model
        self._term = term

    def __dir__(self, forward: bool = True):
        return chain(self._model.__dir__(), self._term.api()) if forward else self._term.api()

    def __getattr__(self, name, forward: bool = True):
        err = AttributeError(
            f"'{self._term.__class__.__name__}' object has no attribute '{name}'", name=name, obj=self._term
        )
        if name in self._term.api():
            return getattr(self._term, name)
        elif forward:
            try:
                return getattr(self._model, name)
            except AttributeError as e:
                raise err from e
        else:
            raise err
