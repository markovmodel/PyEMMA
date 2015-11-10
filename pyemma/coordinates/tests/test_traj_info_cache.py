
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
Created on 30.04.2015

@author: marscher
'''

from __future__ import absolute_import

import os
import tempfile
import unittest

import mdtraj
import pyemma
from pyemma.coordinates.data.traj_info_cache import _TrajectoryInfoCache as TrajectoryInfoCache
from pyemma.util.files import TemporaryDirectory
from pyemma.datasets import get_bpti_test_data

import numpy as np


# os.path.join(path, 'bpti_mini.xtc')
xtcfiles = get_bpti_test_data()['trajs']
pdbfile = get_bpti_test_data()['top']


class TestTrajectoryInfoCache(unittest.TestCase):

    def setUp(self):
        self.tmpfile = tempfile.mktemp()
        self.db = TrajectoryInfoCache(self.tmpfile)
        
    def tearDown(self):
        del self.db
        #os.unlink(self.tmpfile)

    def testCacheResults(self):
        # cause cache failures
        results = {}
        for f in xtcfiles:
            results[f] = self.db[f]

        desired = {}
        for f in xtcfiles:
            with mdtraj.open(f) as fh:
                desired[f] = len(fh)

        self.assertEqual(results, desired)

    def test_with_npy_file(self):
        from pyemma.util.files import TemporaryDirectory
        lengths = [1, 23, 27, ]
        different_lengths_array = [np.empty((n, 3)) for n in lengths]
        files = []
        with TemporaryDirectory() as td:
            for i, x in enumerate(different_lengths_array):
                fn = os.path.join(td, "%i.npy" % i)
                np.save(fn, x)
                files.append(fn)

            # cache it and compare
            results = {f: self.db[f] for f in files}
            expected = {fn: len(different_lengths_array[i]) for i, fn in enumerate(files)}

            self.assertEqual(results, expected)
            
    def test_xyz(self):
        traj = mdtraj.load(xtcfiles, top=pdbfile)
        expected = len(traj)
        import warnings
        from pyemma.util.exceptions import EfficiencyWarning

        with TemporaryDirectory() as td:
            fn = os.path.join(td, 'test.xyz')
            traj.save_xyz(fn)
            with warnings.catch_warnings(record=True) as w:
                reader = pyemma.coordinates.source(fn, top=pdbfile)
                
                self.assertEqual(len(w), 1)
                self.assertEqual(w[0].category, EfficiencyWarning)
                
            self.assertEqual(reader.trajectory_length(0), expected)
            

if __name__ == "__main__":
    unittest.main()
