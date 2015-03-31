'''
Created on 23.01.2015

@author: marscher
'''
import mdtraj
import os
import tempfile
import unittest

from pyemma.coordinates.io.feature_reader import FeatureReader
from pyemma.util.log import getLogger
import pkg_resources

import numpy as np
from pyemma.coordinates.api import feature_reader, discretizer, tica

log = getLogger('TestFeatureReader')


class TestFeatureReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        c = super(TestFeatureReader, cls).setUpClass()
        # create a fake trajectory which has 3 atoms and coordinates are just a range
        # over all frames.
        cls.trajfile = tempfile.mktemp('.xtc')
        cls.n_frames = 1000
        cls.xyz = np.random.random(cls.n_frames * 3 * 3).reshape((cls.n_frames, 3, 3))
        log.debug("shape traj: %s" % str(cls.xyz.shape))
        cls.topfile = pkg_resources.resource_filename(
            'pyemma.coordinates.tests.test_featurereader', 'data/test.pdb')
        t = mdtraj.load(cls.topfile)
        t.xyz = cls.xyz
        t.time = np.arange(cls.n_frames)
        t.save(cls.trajfile)
        return c

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.trajfile)
        super(TestFeatureReader, cls).tearDownClass()

    def testIteratorAccess(self):
        reader = FeatureReader(self.trajfile, self.topfile)

        frames = 0
        data = []
        for i, X in reader:
            frames += X.shape[0]
            data.append(X)

        # restore shape of input
        data = np.array(data).reshape(self.xyz.shape)

        self.assertEqual(frames, reader.trajectory_lengths()[0])
        self.assertTrue(np.allclose(data, self.xyz))

    def testIteratorAccess2(self):
        reader = FeatureReader([self.trajfile, self.trajfile], self.topfile)
        reader.chunksize = 100

        frames = 0
        data = []
        for i, X in reader:
            frames += X.shape[0]
            data.append(X)
        self.assertEqual(frames, reader.trajectory_lengths()[0] * 2)
        # restore shape of input
        data = np.array(
            data[0:reader.trajectory_lengths()[0] / reader.chunksize]).reshape(self.xyz.shape)

        self.assertTrue(np.allclose(data, self.xyz))

    def testTimeLaggedIterator(self):
        lag = 10
        reader = FeatureReader(self.trajfile, self.topfile)
        reader.lag = lag
        frames = 0
        data = []
        lagged = []
        for _, X, Y in reader:
            frames += X.shape[0]
            data.append(X)
            lagged.append(Y)

        assert len(data) == len(lagged)
        merged_lagged = np.concatenate(lagged,axis=0)  # .reshape(self.xyz.shape)

        # reproduce outcome
        xyz_s = self.xyz.shape
        fake_lagged = np.empty((xyz_s[0]-lag,xyz_s[1]*xyz_s[2]))
        fake_lagged = self.xyz.reshape((xyz_s[0],-1))[lag:]

        self.assertTrue(np.allclose(merged_lagged, fake_lagged))

        # restore shape of input
        data = np.array(data).reshape(self.xyz.shape)

        self.assertEqual(frames, reader.trajectory_lengths()[0])
        self.assertTrue(np.allclose(data, self.xyz))
        
    def test_with_pipeline_time_lagged(self):
        reader= feature_reader(self.trajfile, self.topfile)
        reader.featurizer.distances([[0,1], [0,2]])
        t = tica(dim=2,lag=1)
        d = discretizer(reader, t)
        d.parametrize()

    #@unittest.skip("")
    def testTimeLaggedAccess(self):
        # each frame has 2 atoms with 3 coords = 6 coords per frame.
        # coords are sequential through all frames and start with 0.

        lags = [2, 200]

        chunksizes = [1, 100]

        for lag in lags:
            for chunksize in chunksizes:
                log.info("chunksize=%i\tlag=%i" % (chunksize, lag))

                lagged_chunks = []
                reader = FeatureReader(self.trajfile, self.topfile)
                reader.chunksize = chunksize
                reader.lag = lag
                for _, _, y in reader:
                    lagged_chunks.append(y)

                coords = self.xyz.reshape((self.xyz.shape[0],-1))

                for ii, c in enumerate(lagged_chunks[:-1]):
                    # all despite last chunk shall have chunksize
                    self.assertTrue(c.shape[0] <= chunksize)
                    # first lagged chunk should start at lag and stop at chunksize +
                    # lag
                    ind1 = ii * chunksize + lag
                    ind2 = ind1 + chunksize
                    #log.debug("coor slice[%i: %i]" % (ind1, ind2))
                    np.testing.assert_allclose(c, coords[ind1:ind2])

                # TODO: check last lagged frame

                # last lagged chunk should miss "lag" frames of input! e.g
                # padded to maintain chunksize

                last_chunk = lagged_chunks[-1]
                # print last_chunk
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
