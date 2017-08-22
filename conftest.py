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
        pyemma.config.use_trajectory_lengths_cache = False
    yield
