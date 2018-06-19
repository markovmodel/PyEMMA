import pytest
import sys

# this is import for measuring coverage
assert 'pyemma' not in sys.modules


@pytest.fixture(scope='session')
def no_progress_bars():
    """ disables progress bars during testing """
    if 'pyemma' in sys.modules:
        pyemma = sys.modules['pyemma']
        pyemma.config.show_progress_bars = False
        pyemma.config.coordinates_check_output = True
        pyemma.config.use_trajectory_lengths_cache = False
    yield


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    # we enforce legacy string formatting of numpy arrays, because the output format changed in version 1.14,
    # leading to failing doctests.
    import numpy as np
    try:
        np.set_printoptions(legacy='1.13')
    except TypeError:
        pass


@pytest.fixture(autouse=True)
def filter_warnings():
    import warnings
    old_filters = warnings.filters[:]
    warnings.filterwarnings('ignore', message='You have not selected any features. Returning plain coordinates.')
    yield
    warnings.filters = old_filters
