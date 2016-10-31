import pytest


@pytest.fixture(scope='session')
def no_progress_bars():
    """ disables progress bars during testing """
    import pyemma
    pyemma.config.show_progress_bars = False


def static_var(name, value):
    def decorate(func):
        setattr(func, name, value)
        return func
    return decorate


def get_extension_names():
    import json, pkg_resources
    extensions_file = pkg_resources.resource_filename('pyemma', '_extensions.json')
    with open(extensions_file) as f:
        inp = json.load(f)
    extension_stubs = [e.split('.')[-1] for e in inp]
    return extension_stubs


# this hook should ignore setuptools generated stub loaders for extension,
# because they cause PathMismatch errors in pytest during discovery.
@static_var('extension_stubs', get_extension_names())
def pytest_ignore_collect(path, config):

    if any(path.basename.startswith(e) for e in pytest_ignore_collect.extension_stubs):
        return True
