import numpy as np
import pytest
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

from mofaflex._core.likelihoods import Likelihood


@pytest.fixture(scope="module", params=["Normal", "Bernoulli", "NegativeBinomial"])
def likelihood(request):
    return request.param


@pytest.fixture(scope="module", params=[np.asarray, csc_array, csc_matrix, csr_array, csr_matrix])
def adata(rng, create_adata, likelihood, random_array, request):
    return create_adata(request.param(random_array(likelihood, (20, 5))))


def test_infer_likelihoods(adata, likelihood):
    inferred = Likelihood.infer(adata)
    assert likelihood == inferred.__name__


def test_validate_likelihoods(adata, likelihood):
    Likelihood.known_likelihoods()[likelihood].validate(adata, None, None)
