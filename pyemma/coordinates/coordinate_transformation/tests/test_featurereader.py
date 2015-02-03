'''
Created on 23.01.2015

@author: marscher
'''
import pkg_resources
import tempfile
import unittest

from pyemma.util.log import getLogger
import mdtraj

from pyemma.coordinates.coordinate_transformation.io.feature_reader import FeatureReader
from pyemma.coordinates.coordinate_transformation.io.featurizer import MDFeaturizer
from pyemma.coordinates.coordinate_transformation.transform.transformer import Transformer
import numpy as np
import os


log = getLogger('TestFeatureReader')


def map_return_input(traj):
    return traj.xyz


class MemoryStorage(Transformer):
    """stores added data in memory (list of ndarrays) """

    def __init__(self, chunksize=100, lag=0):
        Transformer.__init__(self, chunksize, lag)

        self.lagged_chunks = []

    def param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None):
        self.lagged_chunks.append(Y)
        if last_chunk:
            return True


class TestFeatureReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        c = super(TestFeatureReader, cls).setUpClass()
        # create a fake trajectory which has 2 atoms and coordinates are just a range
        # over all frames.
        cls.trajfile = tempfile.mktemp('.xtc')
        cls.n_frames = 1000
        cls.xyz = np.arange(cls.n_frames * 2 * 3).reshape((cls.n_frames, 2, 3))
        cls.topfile = pkg_resources.resource_filename(
            'pyemma.coordinates.coordinate_transformation.tests.test_featurereader', 'data/test.pdb')
        t = mdtraj.load(cls.topfile)
        t.xyz = cls.xyz
        t.time = np.arange(cls.n_frames)
        t.save(cls.trajfile)
        return c

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.trajfile)
        super(TestFeatureReader, cls).tearDownClass()

    def testTimeLaggedAccess(self):
        # each frame has 2 atoms with 3 coords = 6 coords per frame.
        # coords are sequential through all frames and start with 0.
        trajfiles = [self.trajfile]

        # so first lagged frame is 18, since 3 frames a 6 coords are skipped
        lags = [1, 3, 5, 10, 29, 100]

        #chunksizes = [2, 3, 7, 11, 10, 101, 512]
        chunksizes = [2, 4, 16, 24, 48, 64, 127]

        for lag in lags:
            for chunksize in chunksizes:
                if chunksize <= lag:
                    continue
                log.info("chunksize=%i\tlag=%i" % (chunksize, lag))
                f = MDFeaturizer(self.topfile)
                f.map = map_return_input

                reader = FeatureReader(trajfiles, self.topfile)
                reader.feature = f

                m = MemoryStorage(lag=lag)
                m.data_producer = reader
                chain = [f, reader, m]
                for t in chain:
                    t.chunksize = chunksize
                m.parametrize()

                coords = self.xyz

                # all despite last chunk shall have chunksize
                for ii, c in enumerate(m.lagged_chunks[:-1]):
                    #log.debug("ind: %i" % ii)
                    self.assertEqual(c.shape[0], chunksize)
                    # first lagged chunk should start at lag and stop at chunksize +
                    # lag
                    ind1 = ii * chunksize + lag
                    ind2 = ind1 + chunksize
                    #log.debug("coor slice[%i: %i]" % (ind1, ind2))
                    np.testing.assert_allclose(c, coords[ind1:ind2])

                # TODO: check last lagged frame

                # last lagged chunk should miss "lag" frames of input! e.g
                # padded to maintain chunksize

                last_chunk = m.lagged_chunks[-1]
                # when is last_chunk padded?
                # if 
                # how many zeros are going to be used?


#                 expected = np.empty((chunksize, 2, 3))
#                 for ii, c in enumerate(xrange(chunksize)):
#                     c += 1
#                     expected[ii] = coords[-c]
# expected  = np.array((coords[-2], coords[-1]))
#                 print last_chunk
#                 print "-"*10
#                 print expected
                #np.testing.assert_allclose(last_chunk, expected)

if __name__ == "__main__":
    unittest.main()
