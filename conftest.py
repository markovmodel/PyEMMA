import pytest
import sys
import os

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


def pytest_collection_modifyitems(session, config, items):

    circle_node_total, circle_node_index = read_circleci_env_variables()
    deselected = []
    # round robbin:
    # first group by parent, then skip by ci node index
    #from collections import defaultdict
    #by_parents = defaultdict(list)
    #for index, item in enumerate(items):
    #    by_parents[item.parent].append(item)
    for item in items:
        print(item.location)
        i = hash(item.name)
        if i % circle_node_total != circle_node_index:
            deselected.append(i)
            items.remove(item)
    config.hook.pytest_deselected(items=deselected)


def read_circleci_env_variables():
    """Read and convert CIRCLE_* environment variables"""
    circle_node_total = int(os.environ.get("CIRCLE_NODE_TOTAL", "1").strip() or "1")
    circle_node_index = int(os.environ.get("CIRCLE_NODE_INDEX", "0").strip() or "0")

    if circle_node_index >= circle_node_total:
        raise RuntimeError("CIRCLE_NODE_INDEX={} >= CIRCLE_NODE_TOTAL={}, should be less".format(circle_node_index, circle_node_total))

    return circle_node_total, circle_node_index


def pytest_report_header(config):
    """Add CircleCI information to report"""
    circle_node_total, circle_node_index = read_circleci_env_variables()
    return "CircleCI total nodes: {}, this node index: {}".format(circle_node_total, circle_node_index)
