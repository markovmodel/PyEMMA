import pytest


@pytest.fixture(scope='session')
def no_progress_bars():
    """ disables progress bars during testing """
    import pyemma
    pyemma.config.show_progress_bars = False
    pyemma.config.use_trajectory_lengths_cache = False
