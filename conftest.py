import pytest
import sys
import os

# this is import for measuring coverage
assert 'pyemma' not in sys.modules


def setup_pyemma_config():
    """ set config flags for testing """
    if 'pyemma' in sys.modules:
        pyemma = sys.modules['pyemma']
        pyemma.config.show_progress_bars = False
        pyemma.config.coordinates_check_output = True
        pyemma.config.use_trajectory_lengths_cache = False


def add_np():
    # we enforce legacy string formatting of numpy arrays, because the output format changed in version 1.14,
    # leading to failing doctests.
    import numpy as np
    try:
        np.set_printoptions(legacy='1.13')
    except TypeError:
        pass


@pytest.fixture(scope='session')
def session_fixture():
    setup_pyemma_config()
    add_np()

    # redirect tempdir to a subdir called pyemma-test-$random to clean all temporary files after testing.
    import tempfile, uuid
    org = tempfile.gettempdir()
    tempfile.tempdir = os.path.join(org, 'pyemma_test-{}'.format(uuid.uuid4()))
    print('session temporary dir:', tempfile.tempdir)
    try:
        os.mkdir(tempfile.tempdir)
    except OSError as ose:
        if 'exists' not in ose.strerror.lower():
            raise
    yield
    import shutil
    shutil.rmtree(tempfile.tempdir, ignore_errors=True)
