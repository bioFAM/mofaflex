import pytest

from mofaflex._core.priors import APIType, Prior
from mofaflex._core.utils import MeanStd


class DummyPrior(Prior):
    _factors = True
    _weights = False
    _state_attrs = ("_prop", "_meanstdprop")

    subset_prop = slice(5, 10)

    def __init__(self, names, **kwargs):
        super().__init__(names)
        self._prop = 6.6743e-11
        self._meanstdprop = MeanStd({"test": 2.7182818284}, {"test": 1.6180339887})
        self._customsave = 1.602176634e-19

    @Prior._api
    @property
    def prop_nofactors(self):
        return 42

    @Prior._api(has_factors=True)
    @property
    def prop_factors(self):
        return 1337

    @Prior._api(has_factors=True, factors_subset="subset_prop")
    @property
    def prop_factors_subset(self):
        return 299792458

    @Prior._api(has_factors=False)
    def method_nofactors(self):
        return 3.1415926535

    @Prior._api
    def method_factors(self):
        return 6.62607015e-34

    @Prior._api(factors_subset="subset_prop")
    def method_factors_subset(self):
        return 1.380649e-23

    def posterior(self):
        pass

    def _save(self):
        return {"customsave": self._customsave}

    def _load(self, state, map_location):
        self._customsave = state["customsave"]


@pytest.fixture
def dummyprior():
    return DummyPrior(None)


def test_api():
    for api in DummyPrior.api():
        if api.name == "prop_nofactors":
            assert api.type == APIType.property
            assert api.has_factors is False
            assert api.factors_subset is None
        elif api.name == "prop_factors":
            assert api.type == APIType.property
            assert api.has_factors is True
            assert api.factors_subset is None
        elif api.name == "prop_factors_subset":
            assert api.type == APIType.property
            assert api.has_factors is True
            assert api.factors_subset == "subset_prop"
        elif api.name == "method_nofactors":
            assert api.type == APIType.method
            assert api.has_factors is False
            assert api.factors_subset is None
        elif api.name == "method_factors":
            assert api.type == APIType.method
            assert api.has_factors is True
            assert api.factors_subset is None
        elif api.name == "method_factors_subset":
            assert api.type == APIType.method
            assert api.has_factors is True
            assert api.factors_subset == "subset_prop"

    for meth in DummyPrior.api_methods():
        assert meth.type == APIType.method

    for prop in DummyPrior.api_properties():
        assert prop.type == APIType.property


def test_saveload(dummyprior):
    state = dummyprior.save()
    loaded = DummyPrior.load(state)

    assert dummyprior._prop == loaded._prop
    assert dummyprior._meanstdprop == loaded._meanstdprop
    assert dummyprior._customsave == loaded._customsave
