import inspect
from functools import update_wrapper

from pydocstring import Docstring, Parameter, Section, SectionKind, emit_google, parse_google

import mofaflex as mfl
from mofaflex._core.api.utils import APIType
from mofaflex._core.terms import MofaFlex
from mofaflex._core.api import types


def get_doc(obj) -> Docstring:
    doc = inspect.getdoc(obj)
    if doc is None:
        doc = Docstring()
    else:
        doc = parse_google(doc).to_model()
    return doc


def setup(app):
    # prior API wrapper admonitions
    for apicls, apiclsname in ((getattr(mfl.priors, cls), cls) for cls in dir(mfl.priors) if cls != "Prior"):
        implcls = getattr(mfl._core.priors, apiclsname)
        doc = get_doc(apicls)
        msgs = ""
        if (f := implcls.factors_allowed()) != implcls.weights_allowed():
            msgs += f".. important::\n   This prior can only be used for {'factors' if f else 'weights'}.\n\n"
        if len(implcls.api()) > 0:
            msgs += ".. important::\n   All methods and properties of this class are only accessible through the :class:`~.terms.MofaFlex` class.\n\n"

        if doc.extended_summary is None:
            doc.extended_summary = msgs
        else:
            doc.extended_summary = f"{msgs}{doc.extended_summary}"
        apicls.__doc__ = emit_google(doc)

    # MofaFlex term dynamic api
    apinames = mfl._core.terms.mofaflex._apinames
    getters = MofaFlex.get_factors, MofaFlex.get_weights
    for axis, priors in enumerate(
        (mfl._core.priors.Prior.known_priors("factors"), mfl._core.priors.Prior.known_priors("weights"))
    ):
        getter_doc = get_doc(getters[axis])
        getter_signature_params = inspect.signature(getters[axis]).parameters
        getter_params = next(
            section.parameters for section in getter_doc.sections if section.kind == SectionKind.PARAMETERS
        )
        seen_getter_params = {param.names[0] for param in getter_params}

        for prior, priorcls in priors.items():
            for api in priorcls.api():
                name = apinames[(axis, prior, api.name)]
                wrappedapi = getattr(MofaFlex, name)
                doc = get_doc(wrappedapi)
                desc = doc.extended_summary or ""
                if api.type == APIType.property and not api.has_factors:
                    desc = f".. important::\n   This property is only available when using the :class:`~.priors.{prior}` prior.\n\n{desc}"

                    setattr(getattr(mfl.priors, prior), name, getattr(priorcls, api.name))
                else:
                    desc = f".. important::\n   This method is only available when using the :class:`~.priors.{prior}` prior.\n\n{desc}"
                    if api.has_factors:
                        param = Parameter(
                            names=["ordered"],
                            description="Whether to return the factors ordered by explained variance (highest to lowest).",
                        )
                        for section in doc.sections:
                            if section.kind == SectionKind.PARAMETERS:
                                section.parameters.append(param)
                                break
                        else:
                            doc.sections.append(Section(SectionKind.PARAMETERS, parameters=[param]))

                    wrapper2 = lambda self, *args, **kwargs: None
                    wrapper2.__signature__ = wrappedapi.__signature__
                    update_wrapper(wrapper2, wrappedapi)
                    wrapper2.__doc__ = emit_google(doc)
                    setattr(getattr(mfl._core.api.priors, prior), name, wrapper2)

                if len(desc) > 0:
                    doc.extended_summary = desc
                wrappedapi.__doc__ = emit_google(doc)

            postprocess_doc = get_doc(priorcls.postprocess_results)
            if params := next(
                (section.parameters for section in postprocess_doc.sections if section.kind == SectionKind.PARAMETERS),
                None,
            ):
                for param in params:
                    if param.names[0] not in seen_getter_params and param.names[0] in getter_signature_params:
                        desc = param.description
                        if desc is None:
                            desc = ""
                        desc += f"\n\n.. important::\n   This argument is only available when using the :class:`~.priors.{prior}` prior."
                        param.description = desc
                        getter_params.append(param)
        getters[axis].__doc__ = emit_google(getter_doc)

    # terms
    for apicls, apiclsname in ((getattr(mfl.terms, cls), cls) for cls in dir(mfl.terms)):
        implcls = getattr(mfl._core.terms, apiclsname)
        wrapper = type(apiclsname, (), {"__module__": apicls.__module__, "__doc__": implcls.__doc__})
        wrapper.__init__ = implcls.__init__
        for api in implcls.api():
            setattr(wrapper, api.name, getattr(implcls, api.name))
        setattr(mfl.terms, apiclsname, wrapper)
        setattr(types.terms, apiclsname, wrapper)
    types.terms.Term = None

    # likelihoods
    for apicls, apiclsname in (
        (getattr(mfl.likelihoods, cls), cls) for cls in dir(mfl.likelihoods) if cls != "Likelihood"
    ):
        implcls = getattr(mfl._core.likelihoods, apiclsname)
        for api in implcls.api():
            setattr(apicls, api.name, getattr(implcls, api.name))
        setattr(types.likelihoods, apiclsname, apicls)
    types.likelihoods.Likelihood = None
