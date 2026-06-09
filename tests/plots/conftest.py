import inspect
from functools import partial, wraps

import matplotlib.pyplot as plt
import plotnine
import pytest
from matplotlib.testing.decorators import image_comparison as mpl_image_comparison
from packaging.version import Version

image_comparison = partial(
    mpl_image_comparison, extensions=["png"], tol=0.5
)  # tolerance for differences in text rendering


def plotnine_test(i: int, baseline_image: str):
    def wrapper(func):

        plots = None

        @wraps(func)
        def decorated(*args, **kwargs):
            nonlocal plots
            if plots is None:
                plots = func(*args, **kwargs)
                if not isinstance(plots, list | tuple):
                    plots = (plots,)

            plot = plots[i]
            plt.figure(plot.draw(show=False))

        # Workaround: The image_comparison decorator determines the directory with baseline and result images
        # automatically based on the file where the function was defined. Not replacing the filename results
        # in all plots being written to baseline_images/conftest instead of baseline_images/test_plots
        decorated.__code__ = decorated.__code__.replace(co_filename=func.__code__.co_filename)

        # Workaround for a bug in the image_comparison decorator: It stacks functools.wraps on top
        # of pytest decorators. functools.wraps updates __dict__. However, since all marks are in
        # the same attribute, the new marks get overwritten with the old marks.
        del decorated.__dict__["pytestmark"]

        decorated = image_comparison(baseline_images=[baseline_image])(decorated)
        decorated = pytest.mark.xfail(
            condition=Version(plotnine.__version__).is_prerelease, reason="plotnine pre-release", strict=False
        )(decorated)

        decorated.pytestmark += func.pytestmark

        return decorated

    return wrapper


def pytest_pycollect_makeitem(collector, name, obj):
    if collector.istestfunction(obj, name) and hasattr(obj, "pytestmark"):
        frame = inspect.currentframe().f_back
        while frame.f_back is not None:
            if frame.f_code.co_filename == __file__:  # called recursively
                return
            frame = frame.f_back

        for mark in obj.pytestmark:
            if mark.name == "baseline_images":
                items = []

                for i, baseline_image in enumerate(mark.args):
                    cfunc = plotnine_test(i, baseline_image)(obj)

                    for item in collector.ihook.pytest_pycollect_makeitem(collector=collector, name=name, obj=cfunc):
                        item.name += f"[{baseline_image}]"
                        item.obj = cfunc
                        item.own_markers = cfunc.pytestmark
                        items.append(item)
                return items
    return


def pytest_configure(config):
    config.addinivalue_line("markers", "baseline_images")
