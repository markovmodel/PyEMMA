import pytest


@pytest.fixture(scope='session')
def no_progress_bars():
    """ disables progress bars during testing """
    import subprocess

    def pg_enabled():
        out = subprocess.check_output(['python', '-c', 'from __future__ import print_function; import pyemma; print(pyemma.config.show_progress_bars)'])
        return out.find(b'True') != -1

    def cache_enabled():
        out = subprocess.check_output(['python', '-c', 'from __future__ import print_function; import pyemma; print(pyemma.config.use_trajectory_lengths_cache)'])
        return out.find(b'True') != -1

    cfg_script = "import pyemma; pyemma.config.show_progress_bars = {pg}; pyemma.config.use_trajectory_lengths_cache = {cache};pyemma.config.save()"

    pg_old_state = pg_enabled()
    cache_old_state = cache_enabled()

    enable = cfg_script.format(pg=pg_old_state, cache=cache_old_state)
    disable = cfg_script.format(pg=False, cache=False)

    subprocess.call(['python', '-c', disable])
    yield  # run session, after generator returned, session is cleaned up.
    subprocess.call(['python', '-c', enable])
