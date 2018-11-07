import sys
import warnings
from contextlib import contextmanager

import pyemma
import unittest
import pkg_resources
import os

data_path = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/serialization')


def get_path(fn):
    return os.path.join(data_path, fn)


@contextmanager
def supress_deprecation_warning():
    old_filters = warnings.filters[:]
    from pyemma.util.exceptions import PyEMMA_DeprecationWarning
    warnings.filterwarnings('ignore', category=PyEMMA_DeprecationWarning)

    yield
    warnings.filters = old_filters


@unittest.skipIf(sys.version_info[0] < 3, 'py3 only')
class TestTICARestorePriorVersions(unittest.TestCase):

    def test_default_values(self):
        t = pyemma.load(get_path('tica_2.5.4.pyemma'))
        assert t.scaling == 'kinetic_map'
        assert t.dim == 0.95

    def test_commute_map(self):
        t = pyemma.load(get_path('tica_2.5.2_commute_map.pyemma'))
        assert t.scaling == 'commute_map'
        with supress_deprecation_warning():
            assert t.commute_map
            assert not t.kinetic_map

    def test_fixed_dim(self):
        """ stored model with dim=2"""
        t = pyemma.load(get_path('tica_2.5.2_fixed_dim.pyemma'))
        assert t.dim == 2
        with supress_deprecation_warning():
            assert t.var_cutoff == 1.0

    def test_var_cutoff2(self):
        """ stored model with var_cutoff=0.5"""
        t = pyemma.load(get_path('tica_2.5.2_var_cutoff.pyemma'))
        assert t.dim == 0.5


if __name__ == '__main__':
    unittest.main()
