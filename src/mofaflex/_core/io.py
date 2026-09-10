from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import h5py
from packaging.version import Version

logger = logging.getLogger(__name__)


def save_model(model_state, path: str | Path):
    """Save a MOFA-FLEX model to an HDF5 file.

    Saves both the model state and parameters, with optional MOFA-compatible format.

    Args:
        model_state: The internal state of the model. Should be compatible with `anndata.io.write_elem`.
        path: File path where to save the model.
    """
    from .. import __version__

    dset_kwargs = {"compression": "gzip", "compression_opts": 9}

    path = Path(path)
    if path.exists():
        logger.warning(f"{path} already exists, overwriting")
    with h5py.File(path, "w") as f:
        with ad.settings.override(allow_write_nullable_strings=True):
            ad.io.write_elem(
                f,
                "mofaflex",
                model_state,
                dataset_kwargs={} if Version(ad.__version__) < Version("0.11.2") else dset_kwargs,
            )  # https://github.com/h5py/h5py/issues/2525

        f["mofaflex"].attrs["version"] = __version__


def load_model(path: str | Path):
    """Load a MOFA-FLEX model from an HDF5 file.

    Args:
        path: Path to the HDF5 file containing the saved model.
        map_location: Optional device specification for loading the model.

    Returns:
        The loaded MOFA-FLEX model.
    """
    from .. import __version__

    path = Path(path)
    with h5py.File(path, "r") as f:
        mofaflexgrp = f["mofaflex"]
        version = mofaflexgrp.attrs["version"]
        if Version(version).__replace__(pre=None, post=None, dev=None, local=None) < Version("0.2"):
            raise ValueError(
                f"The stored model was created with MOFA-FLEX 0.1.x and cannot be loaded by MOFA-FLEX {__version__}."
            )
        if version != __version__:
            logger.warning(
                "The stored model was created with a different version of MOFA-FLEX. Some features may not work."
            )
        state = ad.io.read_elem(mofaflexgrp)

    return state
