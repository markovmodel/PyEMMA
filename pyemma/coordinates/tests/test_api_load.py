
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on 14.04.2015

@author: marscher
'''

from __future__ import absolute_import
# unicode compat py2/3
from six import text_type
import unittest
from pyemma.coordinates.api import load
import os

import numpy as np
from pyemma.coordinates import api

import pkg_resources
path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep

pdb_file = os.path.join(path, 'bpti_ca.pdb')
traj_files = [
    os.path.join(path, 'bpti_001-033.xtc'),
    os.path.join(path, 'bpti_067-100.xtc')
]


class TestAPILoad(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
        cls.bpti_pdbfile = os.path.join(path, 'bpti_ca.pdb')
        extensions = ['.xtc', '.binpos', '.dcd', '.h5', '.lh5', '.nc', '.netcdf', '.trr']
        cls.bpti_mini_files = [os.path.join(path, 'bpti_mini%s' % ext) for ext in extensions]

    def testUnicodeString_without_featurizer(self):
        filename = text_type(traj_files[0])

        with self.assertRaises(ValueError):
            load(filename)

    def testUnicodeString(self):
        filename = text_type(traj_files[0])
        features = api.featurizer(pdb_file)

        load(filename, features)

    def test_various_formats_load(self):
        chunksizes = [0, 13]
        X = None
        bpti_mini_previous = None
        for cs in chunksizes:
            for bpti_mini in self.bpti_mini_files:
                Y = api.load(bpti_mini, top=self.bpti_pdbfile, chunksize=cs)
                if X is not None:
                    np.testing.assert_array_almost_equal(X, Y, err_msg='Comparing %s to %s failed for chunksize %s'
                                                                       % (bpti_mini, bpti_mini_previous, cs))
                X = Y
                bpti_mini_previous = bpti_mini

    def test_load_traj(self):
        filename = traj_files[0]
        features = api.featurizer(pdb_file)
        res = load(filename, features)

        self.assertEqual(type(res), np.ndarray)

    def test_load_trajs(self):
        features = api.featurizer(pdb_file)
        res = load(traj_files, features)

        self.assertEqual(type(res), list)
        self.assertTrue(all(type(x) == np.ndarray for x in res))

    def test_with_trajs_without_featurizer_or_top(self):

        with self.assertRaises(ValueError):
            load(traj_files)

        output = load(traj_files, top=pdb_file)
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), len(traj_files))

    def test_non_existant_input(self):
        input_files = [traj_files[0], "does_not_exist_for_sure"]

        with self.assertRaises(ValueError):
            load(trajfiles=input_files, top=pdb_file)

    def test_empty_list(self):
        with self.assertRaises(ValueError):
            load([], top=pdb_file)

    def test_load_features_pass_trajectory_as_topology(self):
        import mdtraj
        t = mdtraj.load(pdb_file)
        features = api.featurizer(t)

        traj_files = [f for f in self.bpti_mini_files if f.endswith('.netcdf')]
        load(traj_files, features=features)

if __name__ == "__main__":
    unittest.main()