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
Created on 28.01.2015

@author: marscher
'''

from __future__ import absolute_import

import os
import unittest

from pyemma.coordinates.api import cluster_kmeans
from pyemma.util.files import TemporaryDirectory

from six.moves import range
import numpy as np


class TestKmeans(unittest.TestCase):

    def testDtraj(self):
        self.k = 5
        self.dim = 100
        self.data = [np.random.random((30, self.dim)),
                     np.random.random((37, self.dim))]
        self.kmeans = cluster_kmeans(data=self.data, k=self.k, max_iter=100)

        assert self.kmeans.dtrajs[0].dtype == self.kmeans.output_type()

        prefix = "test"
        extension = ".dtraj"
        with TemporaryDirectory() as outdir:
            self.kmeans.save_dtrajs(trajfiles=None, prefix=prefix,
                                    output_dir=outdir, extension=extension)

            names = ["%s_%i%s" % (prefix, i, extension)
                     for i in range(self.kmeans.data_producer.number_of_trajectories())]
            names = [os.path.join(outdir, n) for n in names]

            # check files with given patterns are there
            for f in names:
                os.stat(f)

    def test_3gaussian_1d_singletraj(self):
        # generate 1D data from three gaussians
        X = [np.random.randn(100)-2.0,
             np.random.randn(100),
             np.random.randn(100)+2.0]
        X = np.hstack(X)

        for init_strategy in ['kmeans++', 'uniform']:
            kmeans = cluster_kmeans(X, k=10, init_strategy=init_strategy)
            cc = kmeans.clustercenters
            assert (np.any(cc < 1.0)), "failed for init_strategy=%s" % init_strategy
            assert (np.any((cc > -1.0) * (cc < 1.0))), "failed for init_strategy=%s" % init_strategy
            assert (np.any(cc > -1.0)), "failed for init_strategy=%s" % init_strategy

            # test fixed seed
            km1 = cluster_kmeans(X, k=10, init_strategy=init_strategy, fixed_seed=True)
            km2 = cluster_kmeans(X, k=10, init_strategy=init_strategy, fixed_seed=True)
            np.testing.assert_array_equal(km1.clustercenters, km2.clustercenters,
                                          "should yield same centers with fixed seed")

            # test that not-fixed seed yields different results
            retry, done = 0, False
            while not done and retry < 4:
                try:
                    km3 = cluster_kmeans(X, k=10, init_strategy=init_strategy, fixed_seed=False)
                    self.assertRaises(AssertionError, np.testing.assert_array_equal,
                                      km1.clustercenters, km3.clustercenters)
                    done = True
                except AssertionError:
                    retry += 1
            self.assertTrue(done, 'using a fixed seed compared to a not fixed one made no difference!')

    def test_3gaussian_2d_multitraj(self):
        # generate 1D data from three gaussians
        X1 = np.zeros((100, 2))
        X1[:, 0] = np.random.randn(100)-2.0
        X2 = np.zeros((100, 2))
        X2[:, 0] = np.random.randn(100)
        X3 = np.zeros((100, 2))
        X3[:, 0] = np.random.randn(100)+2.0
        X = [X1, X2, X3]
        kmeans = cluster_kmeans(X, k=10)
        cc = kmeans.clustercenters
        assert(np.any(cc < 1.0))
        assert(np.any((cc > -1.0) * (cc < 1.0)))
        assert(np.any(cc > -1.0))

    def test_kmeans_equilibrium_state(self):
        initial_centers_equilibrium = [np.array([0, 0, 0])]
        X = np.array([
            np.array([1, 1, 1], dtype=np.float32), np.array([1, 1, -1], dtype=np.float32),
            np.array([1, -1, -1], dtype=np.float32), np.array([-1, -1, -1], dtype=np.float32),
            np.array([-1, 1, 1], dtype=np.float32), np.array([-1, -1, 1], dtype=np.float32),
            np.array([-1, 1, -1], dtype=np.float32), np.array([1, -1, 1], dtype=np.float32)
        ])
        kmeans = cluster_kmeans(X, k=1)
        self.assertEqual(1, len(kmeans.clustercenters), 'If k=1, there should be only one output center.')
        msg = 'Type=' + str(type(kmeans)) + '. ' + \
              'In an equilibrium state the resulting centers should not be different from the initial centers.'
        self.assertTrue(np.array_equal(initial_centers_equilibrium[0], kmeans.clustercenters[0]), msg)

    def test_kmeans_convex_hull(self):
        points = [
            [-212129 / 100000, -20411 / 50000, 2887 / 5000],
            [-212129 / 100000, 40827 / 100000, -5773 / 10000],
            [-141419 / 100000, -5103 / 3125, 2887 / 5000],
            [-141419 / 100000, 1 / 50000, -433 / 250],
            [-70709 / 50000, 3 / 100000, 17321 / 10000],
            [-70709 / 50000, 163301 / 100000, -5773 / 10000],
            [-70709 / 100000, -204121 / 100000, -5773 / 10000],
            [-70709 / 100000, -15309 / 12500, -433 / 250],
            [-17677 / 25000, -122471 / 100000, 17321 / 10000],
            [-70707 / 100000, 122477 / 100000, 17321 / 10000],
            [-70707 / 100000, 102063 / 50000, 2887 / 5000],
            [-17677 / 25000, 30619 / 25000, -433 / 250],
            [8839 / 12500, -15309 / 12500, -433 / 250],
            [35357 / 50000, 102063 / 50000, 2887 / 5000],
            [8839 / 12500, -204121 / 100000, -5773 / 10000],
            [70713 / 100000, -122471 / 100000, 17321 / 10000],
            [70713 / 100000, 30619 / 25000, -433 / 250],
            [35357 / 50000, 122477 / 100000, 17321 / 10000],
            [106067 / 50000, -20411 / 50000, 2887 / 5000],
            [141423 / 100000, -5103 / 3125, 2887 / 5000],
            [141423 / 100000, 1 / 50000, -433 / 250],
            [8839 / 6250, 3 / 100000, 17321 / 10000],
            [8839 / 6250, 163301 / 100000, -5773 / 10000],
            [106067 / 50000, 40827 / 100000, -5773 / 10000],
        ]
        kmeans = cluster_kmeans(np.asarray(points, dtype=np.float32), k=1)
        res = kmeans.clustercenters
        # Check hyperplane inequalities. If they are all fulfilled, the center lies within the convex hull.
        self.assertGreaterEqual(np.inner(np.array([-11785060650000, -6804069750000, -4811167325000], dtype=float),
                                         res) + 25000531219381, 0)
        self.assertGreaterEqual(
            np.inner(np.array([-1767759097500, 1020624896250, 721685304875], dtype=float), res) + 3749956484003, 0)
        self.assertGreaterEqual(np.inner(np.array([-70710363900000, -40824418500000, 57734973820000], dtype=float),
                                         res) + 199998509082907, 0)
        self.assertGreaterEqual(np.inner(np.array([70710363900000, 40824418500000, -57734973820000], dtype=float),
                                         res) + 199998705841169, 0)
        self.assertGreaterEqual(np.inner(np.array([70710363900000, -40824995850000, -28867412195000], dtype=float),
                                         res) + 149999651832937, 0)
        self.assertGreaterEqual(np.inner(np.array([-35355181950000, 20412497925000, -28867282787500], dtype=float),
                                         res) + 100001120662259, 0)
        self.assertGreaterEqual(
            np.inner(np.array([23570121300000, 13608139500000, 9622334650000], dtype=float), res) + 49998241292257,
            0)
        self.assertGreaterEqual(np.inner(np.array([0, 577350000, -204125000], dtype=float), res) + 1060651231, 0)
        self.assertGreaterEqual(np.inner(np.array([35355181950000, -20412497925000, 28867282787500], dtype=float),
                                         res) + 99997486799779, 0)
        self.assertGreaterEqual(np.inner(np.array([0, 72168750, 51030625], dtype=float), res) + 176771554, 0)
        self.assertGreaterEqual(np.inner(np.array([0, -288675000, 102062500], dtype=float), res) + 530329843, 0)
        self.assertGreaterEqual(np.inner(np.array([0, 0, 250], dtype=float), res) + 433, 0)
        self.assertGreaterEqual(np.inner(np.array([0, -144337500, -102061250], dtype=float), res) + 353560531, 0)
        self.assertGreaterEqual(np.inner(np.array([0, 0, -10000], dtype=float), res) + 17321, 0)

    def test_with_n_jobs_minrmsd(self):
        kmeans = cluster_kmeans(np.random.rand(500,3), 10, metric='minRMSD')

    def test_skip(self):
        cluster_kmeans(np.random.rand(100, 3), skip=42)

if __name__ == "__main__":
    unittest.main()
