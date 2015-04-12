import unittest
import os
import numpy as np
import tempfile

from pyemma.coordinates.data import MDFeaturizer
from pyemma.util.log import getLogger
import pyemma.coordinates as coor
import pyemma.util.types as types


logger = getLogger('TestReaderUtils')


class TestCluster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestCluster, cls).setUpClass()
        cls.dtraj_dir = tempfile.mkdtemp()

    def setUp(self):
        # generate Gaussian mixture
        means = [np.array([-3,0]),
                 np.array([-1,1]),
                 np.array([0,0]),
                 np.array([1,-1]),
                 np.array([4,2])]
        widths = [np.array([0.3,2]),
                  np.array([0.3,2]),
                  np.array([0.3,2]),
                  np.array([0.3,2]),
                  np.array([0.3,2])]
        # continuous trajectory
        nsample = 1000
        self.T = len(means)*nsample
        self.X = np.zeros((self.T, 2))
        for i in range(len(means)):
            self.X[i*nsample:(i+1)*nsample,0] = widths[i][0] * np.random.randn() + means[i][0]
            self.X[i*nsample:(i+1)*nsample,1] = widths[i][1] * np.random.randn() + means[i][1]
        # cluster in different ways
        self.km = coor.cluster_kmeans(data = self.X, k = 100)
        self.rs = coor.cluster_regspace(data = self.X, dmin=0.5)
        self.rt = coor.cluster_uniform_time(data = self.X, k = 100)
        self.cl = [self.km, self.rs, self.rt]

    def test_chunksize(self):
        for c in self.cl:
            assert types.is_int(c.chunksize)

    def test_clustercenters(self):
        for c in self.cl:
            assert c.clustercenters.shape[0] == c.n_clusters
            assert c.clustercenters.shape[1] == 2

    def test_data_producer(self):
        for c in self.cl:
            assert c.data_producer is not None

    def test_describe(self):
        for c in self.cl:
            desc = c.describe()
            assert types.is_string(desc) or types.is_list_of_string(desc)

    def test_dimension(self):
        for c in self.cl:
            assert types.is_int(c.dimension())
            assert c.dimension() == 1

    def test_dtrajs(self):
        for c in self.cl:
            assert len(c.dtrajs) == 1
            assert c.dtrajs[0].dtype == c.output_type()
            assert len(c.dtrajs[0]) == self.T

    def test_get_output(self):
        for c in self.cl:
            O = c.get_output()
            assert types.is_list(O)
            assert len(O) == 1
            assert types.is_int_matrix(O[0])
            assert O[0].shape[0] == self.T
            assert O[0].shape[1] == 1

    def test_in_memory(self):
        for c in self.cl:
            assert isinstance(c.in_memory, bool)

    def test_iterator(self):
        for c in self.cl:
            for itraj, chunk in c:
                assert types.is_int(itraj)
                assert types.is_int_matrix(chunk)
                assert chunk.shape[0] <= c.chunksize
                assert chunk.shape[1] == c.dimension()

    def test_map(self):
        for c in self.cl:
            Y = c.map(self.X)
            assert Y.shape[0] == self.T
            assert Y.shape[1] == 1
            # test if consistent with get_output
            assert np.allclose(Y, c.get_output()[0])

    def test_n_frames_total(self):
        for c in self.cl:
            c.n_frames_total() == self.T

    def test_number_of_trajectories(self):
        for c in self.cl:
            c.number_of_trajectories() == 1

    def test_output_type(self):
        for c in self.cl:
            assert c.output_type() == np.int64

    def test_parametrize(self):
        for c in self.cl:
            # nothing should happen
            c.parametrize()

    def test_save_dtrajs(self):
        prefix = "test"
        extension = ".dtraj"
        outdir = self.dtraj_dir
        for c in self.cl:
            c.save_dtrajs(trajfiles=None, prefix=prefix, output_dir=outdir, extension=extension)

            names = ["%s_%i%s" % (prefix, i, extension)
                     for i in xrange(c.data_producer.number_of_trajectories())]
            names = [os.path.join(outdir, n) for n in names]

            # check files with given patterns are there
            for f in names:
                os.stat(f)

    def test_trajectory_length(self):
        for c in self.cl:
            assert c.trajectory_length(0) == self.T
            with self.assertRaises(IndexError):
                c.trajectory_length(1)

    def test_trajectory_lengths(self):
        for c in self.cl:
            assert len(c.trajectory_lengths()) == 1
            assert c.trajectory_lengths()[0] == c.trajectory_length(0)


if __name__ == "__main__":
    unittest.main()
