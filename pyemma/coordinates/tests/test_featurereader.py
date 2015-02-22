'''
Created on 23.01.2015

@author: marscher
'''
import mdtraj
import os
import tempfile
import unittest

from pyemma.coordinates.io.feature_reader import FeatureReader
from pyemma.coordinates.transform.transformer import Transformer
from pyemma.util.log import getLogger
import pkg_resources

from pyemma.coordinates.util.chaining import build_chain, run_chain
import numpy as np
import cProfile


log = getLogger('TestFeatureReader')


class TestFeatureReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        c = super(TestFeatureReader, cls).setUpClass()
        # create a fake trajectory which has 2 atoms and coordinates are just a range
        # over all frames.
        cls.trajfile = tempfile.mktemp('.xtc')
        cls.n_frames = 1000
        cls.xyz = np.arange(cls.n_frames * 2 * 3).reshape((cls.n_frames, 2, 3))
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

    def setUp(self):
        self.pr = cProfile.Profile()
        self.pr.enable()

    def tearDown(self):
        from pstats import Stats
        p = Stats(self.pr)
        p.sort_stats('cumtime')
        # p.print_stats()
        p.print_callers('heavy_function')

    def testIteratorAccess(self):
        reader = FeatureReader(self.trajfile, self.topfile)

        frames = 0
        data = []
        for i, X in reader:
            frames += X.xyz.shape[0]
            data.append(X.xyz)

        # restore shape of input
        data = np.array(data).reshape(self.xyz.shape)

        self.assertEqual(frames, reader.trajectory_lengths()[0])
        np.testing.assert_equal(data, self.xyz)

    def testIteratorAccess2(self):
        reader = FeatureReader([self.trajfile, self.trajfile], self.topfile)
        reader.chunksize = 100

        frames = 0
        data = []
        for i, X in reader:
            frames += X.xyz.shape[0]
            data.append(X.xyz)
        self.assertEqual(frames, reader.trajectory_lengths()[0] * 2)
        # restore shape of input
        data = np.array(
            data[0:reader.trajectory_lengths()[0] / reader.chunksize]).reshape(self.xyz.shape)

        np.testing.assert_equal(data, self.xyz)

    def testTimeLaggedIterator(self):
        lag = 10
        reader = FeatureReader(self.trajfile, self.topfile)
        reader.lag = lag
        frames = 0
        data = []
        lagged = []
        for _, X, Y in reader:
            frames += X.xyz.shape[0]
            data.append(X.xyz)
            lagged.append(Y.xyz)

        merged_lagged = np.array(lagged).reshape(self.xyz.shape)

        # reproduce outcome
        fake_lagged = np.empty_like(self.xyz)
        fake_lagged[:-lag] = self.xyz[lag:]
        fake_lagged[-lag:] = 0

        np.testing.assert_equal(merged_lagged, fake_lagged)

        # restore shape of input
        data = np.array(data).reshape(self.xyz.shape)

        self.assertEqual(frames, reader.trajectory_lengths()[0])
        np.testing.assert_equal(data, self.xyz)

    @unittest.skip("")
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
                    lagged_chunks.append(y.xyz)

                coords = self.xyz

                for ii, c in enumerate(lagged_chunks[:-1]):
                    # all despite last chunk shall have chunksize
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
