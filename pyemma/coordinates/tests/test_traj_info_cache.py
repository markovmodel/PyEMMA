'''
Created on 30.04.2015

@author: marscher
'''
import unittest
import os
import tempfile
from glob import glob

from pyemma.coordinates.data.traj_info_cache import _TrajectoryInfoCache as TrajectoryInfoCache
import mdtraj

path = os.path.join(os.path.split(__file__)[0], 'data')
# os.path.join(path, 'bpti_mini.xtc')
xtcfiles = glob(path + os.path.sep + "*.xtc")
pdbfile = os.path.join(path, 'bpti_ca.pdb')


class TestTrajectoryInfoCache(unittest.TestCase):

    def setUp(self):
        self.tmpfile = tempfile.mktemp()
        self.db = TrajectoryInfoCache(self.tmpfile)

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

if __name__ == "__main__":
    unittest.main()
