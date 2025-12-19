from inspect import Parameter, Signature, signature

from ..terms import Term

__all__ = []


def _init_api():
    def make_wrapper(term: Term):  # required due to Python's late-binding closures
        def wrapper(name, /, **kwargs):
            from ..mofaflex import MOFAFLEX

            return MOFAFLEX(**{name: term(**kwargs)})

        return wrapper

    for termname, term in Term.known_terms.items():
        wrapper = make_wrapper(term)
        sig = signature(term.__init__)
        params = [signature(wrapper).parameters["name"]] + [
            Parameter(param.name, Parameter.KEYWORD_ONLY, default=param.default, annotation=param.annotation)
            for param in sig.parameters.values()
        ]
        wrapper.__signature__ = Signature(params)
        wrapper.__annotations__ = term.__init__.__annotations__ | {"name": str}
        wrapper.__doc__ = term.__doc__

        globals()[termname] = wrapper
        __all__.append(termname)


def __dir__():
    return __all__


_init_api()
