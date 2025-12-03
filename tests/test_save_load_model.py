import os
import tempfile
from collections.abc import Mapping, Sequence

import anndata as ad
import numpy as np
import pytest
from packaging.version import Version

from mofaflex import MOFAFLEX, DataOptions, ModelOptions, TrainingOptions


def compare_nested(data1, data2):
    if isinstance(data1, Mapping) and isinstance(data2, Mapping):
        if data1.keys() != data2.keys():
            return False
        return all(compare_nested(data1[k], data2[k]) for k in data1.keys())
    elif (
        isinstance(data1, Sequence | set)
        and not isinstance(data1, str | bytes)
        and isinstance(data2, Sequence | set)
        and not isinstance(data2, str | bytes)
    ):
        if len(data1) != len(data2):
            return False
        return all(compare_nested(d1, d2) for d1, d2 in zip(data1, data2, strict=False))
    elif isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        return np.all(data1 == data2)
    else:
        return data1 == data2


@pytest.fixture
def setup_teardown():
    # Setup: Create a temporary directory for saving models
    temp_file = tempfile.mkstemp(suffix=".h5")
    os.close(temp_file[0])

    yield temp_file[1]

    # Teardown: Remove the temporary directory and its contents
    os.unlink(temp_file[1])


@pytest.mark.xfail(
    Version(ad.__version__) >= Version("0.12.0rc1") and Version(ad.__version__) < Version("0.12.0"),
    reason="anndata bug: https://github.com/scverse/anndata/pull/1975",
    strict=False,
)
def test_save_load_model(setup_teardown):
    temp_file = setup_teardown

    # Prepare dummy data
    data = {"group1": {"view1": ad.AnnData(X=np.random.rand(3, 10)), "view2": ad.AnnData(X=np.random.rand(3, 5))}}

    # Create and train the MOFA-FLEX model for a single epoch
    model = MOFAFLEX(
        data,
        DataOptions(scale_per_group=False, plot_data_overview=False),
        ModelOptions(
            n_factors=2,
            likelihoods={"view1": "Normal", "view2": "Normal"},
            factor_prior="Normal",
            weight_prior="Normal",
        ),
        TrainingOptions(
            device="cpu",
            lr=0.001,
            max_epochs=1,  # Train for a single epoch
            save_path=temp_file,
            mofa_compat="full",
        ),
    )

    # Check if files are saved
    assert os.path.exists(temp_file)

    # Load the model and its parameters
    loaded_model = MOFAFLEX.load(path=temp_file)
    # Check if the model's parameter is correctly loaded
    for attr in (
        "group_names",
        "n_groups",
        "view_names",
        "n_views",
        "feature_names",
        "n_features",
        "sample_names",
        "n_samples",
        "n_samples_total",
        "n_factors",
        "n_total_factors",
        "factor_order",
        "factor_names",
    ):
        assert compare_nested(getattr(model, attr), getattr(loaded_model, attr)), attr
