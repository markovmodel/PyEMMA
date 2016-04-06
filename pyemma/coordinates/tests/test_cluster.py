
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



from __future__ import absolute_import
import unittest
import os
import tempfile

import numpy as np

from logging import getLogger
import pyemma.coordinates as coor
import pyemma.util.types as types
from six.moves import range


logger = getLogger('pyemma.'+'TestReaderUtils')


class TestCluster(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestCluster, cls).setUpClass()
        cls.dtraj_dir = tempfile.mkdtemp()

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
        cls.T = len(means)*nsample
        cls.X = np.zeros((cls.T, 2))
        for i in range(len(means)):
            cls.X[i*nsample:(i+1)*nsample,0] = widths[i][0] * np.random.randn() + means[i][0]
            cls.X[i*nsample:(i+1)*nsample,1] = widths[i][1] * np.random.randn() + means[i][1]
        # cluster in different ways
        cls.km = coor.cluster_kmeans(data = cls.X, k = 100)
        cls.rs = coor.cluster_regspace(data = cls.X, dmin=0.5)
        cls.rt = coor.cluster_uniform_time(data = cls.X, k = 100)
        cls.cl = [cls.km, cls.rs, cls.rt]

    def setUp(self):
        pass

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
                assert chunk.shape[0] <= c.chunksize or c.chunksize == 0
                assert chunk.shape[1] == c.dimension()

    def test_map(self):
        for c in self.cl:
            Y = c.transform(self.X)
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
            assert c.output_type() == np.int32

    def test_parametrize(self):
        for c in self.cl:
            # nothing should happen
            c.parametrize()

    def test_save_dtrajs(self):
        extension = ".dtraj"
        outdir = self.dtraj_dir
        for c in self.cl:
            prefix = "test_save_dtrajs_%s" % type(c).__name__
            c.save_dtrajs(trajfiles=None, prefix=prefix, output_dir=outdir, extension=extension)

            names = ["%s_%i%s" % (prefix, i, extension)
                     for i in range(c.data_producer.number_of_trajectories())]
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