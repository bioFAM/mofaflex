from contextlib import contextmanager
from dataclasses import dataclass


@dataclass(kw_only=True, slots=True)
class Settings:
    """Global settings.

    Examples:
        >>> settings.use_dask = True
        >>> with settings.override(use_dask=False):
        ...     pass
    """

    use_dask: bool = True
    """Whether to use dask where possible to perform operations lazily."""

    dask_chunksize_mb: float = 500
    """Size of Dask array chunks in MiB."""

    eps: float = 1e-8
    """Small epsilon for numerical stability."""

    subsample_size: int = 1000
    """Several pre- and post-training calculations use a random subsample of the data
        to reduce memory and runtime. This setting determines the size of the data sample:
        Larger values yield more precise calculations, but also result in higher memory
        consumption and longer runtime. Set to 0 to disable subsampling and use all data."""

    def set(self, **kwargs):
        """Set settings.

        Args:
            **kwargs: Setting names and their values.
        """
        for argname, argval in kwargs.items():
            setattr(self, argname, argval)

    def get(self, setting: str):
        """Get the current value of a setting.

        Args:
            setting: The name of the settings.
        """
        return getattr(self, setting)

    @contextmanager
    def override(self, **kwargs):
        """Context manager to locally override some settings.

        Args:
            **kwargs: Setting names and their values.
        """
        oldsettings = {}
        for argname in kwargs.keys():
            oldsettings[argname] = getattr(self, argname)
        self.set(**kwargs)
        yield
        self.set(**oldsettings)


settings = Settings()
