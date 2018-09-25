import warnings

import pyemma
import unittest
import pkg_resources
import os

data_path = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/serialization')


def get_path(fn):
    return os.path.join(data_path, fn)


class TestTICARestorePriorVersions(unittest.TestCase):

    def setUp(self):
        from pyemma.util.exceptions import PyEMMA_DeprecationWarning
        self.old_filters = warnings.filters[:]
        warnings.filterwarnings('ignore', category=PyEMMA_DeprecationWarning)

    def tearDown(self):
        warnings.filters = self.old_filters

    def test_default_values(self):
        t = pyemma.load(get_path('tica_2.5.4.pyemma'))
        assert t.scaling == 'kinetic_map'
        assert t.dim == 0.95

    def test_commute_map(self):
        t = pyemma.load(get_path('tica_2.5.2_commute_map.pyemma'))
        assert t.scaling == 'commute_map'
        assert t.commute_map
        assert not t.kinetic_map

    def test_fixed_dim(self):
        """ stored model with dim=2"""
        t = pyemma.load(get_path('tica_2.5.2_fixed_dim.pyemma'))
        assert t.dim == 2
        assert t.var_cutoff == 1.0

    def test_var_cutoff2(self):
        """ stored model with var_cutoff=0.5"""
        t = pyemma.load(get_path('tica_2.5.2_var_cutoff.pyemma'))
        assert t.dim == 0.5

if __name__ == '__main__':
    unittest.main()
