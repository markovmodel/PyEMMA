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
import random
import unittest
import numpy as np

from pyemma.coordinates.api import cluster_kmeans
from pyemma.util.files import TemporaryDirectory


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

        from pyemma.util.contexts import numpy_random_seed
        with numpy_random_seed(42):
            X = [np.random.randn(200)-2.0,
                 np.random.randn(200),
                 np.random.randn(200)+2.0]
        X = np.hstack(X)
        k = 50
        from pyemma._base.estimator import param_grid
        grid = param_grid({'init_strategy': ['uniform', 'kmeans++'], 'fixed_seed': [True, 463498]})
        for param in grid:
            init_strategy = param['init_strategy']
            fixed_seed = param['fixed_seed']
            kmeans = cluster_kmeans(X, k=k, init_strategy=init_strategy, n_jobs=0, fixed_seed=fixed_seed)
            cc = kmeans.clustercenters
            self.assertTrue(np.all(np.isfinite(cc)), "cluster centers borked for strat %s" % init_strategy)
            assert (np.any(cc < 1.0)), "failed for init_strategy=%s" % init_strategy
            assert (np.any((cc > -1.0) * (cc < 1.0))), "failed for init_strategy=%s" % init_strategy
            assert (np.any(cc > -1.0)), "failed for init_strategy=%s" % init_strategy

            km1 = cluster_kmeans(X, k=k, init_strategy=init_strategy, fixed_seed=fixed_seed, n_jobs=0)
            km2 = cluster_kmeans(X, k=k, init_strategy=init_strategy, fixed_seed=fixed_seed, n_jobs=0)
            self.assertEqual(len(km1.clustercenters), k)
            self.assertEqual(len(km2.clustercenters), k)
            self.assertEqual(km1.fixed_seed, km2.fixed_seed)

            # check initial centers (after kmeans++, uniform init) are equal.
            np.testing.assert_equal(km1.initial_centers_, km2.initial_centers_)

            while not km1.converged:
                km1.estimate(X=X, clustercenters=km1.clustercenters, keep_data=True)
            while not km2.converged:
                km2.estimate(X=X, clustercenters=km2.clustercenters, keep_data=True)

            assert np.linalg.norm(km1.clustercenters - km1.initial_centers_) > 0
            np.testing.assert_allclose(km1.clustercenters, km2.clustercenters,
                                       err_msg="should yield same centers with fixed seed=%s for strategy %s, Initial centers=%s"
                                               % (fixed_seed, init_strategy, km2.initial_centers_), atol=1e-6)

    def test_check_convergence_serial_parallel(self):
        """ check serial and parallel version of kmeans converge to the same centers.

        Artificial data set is created with 6 disjoint point blobs, to ensure the parallel and the serial version
        converge to the same result. If the blobs would overlap we can not guarantee this, because the parallel version
        can potentially converge to a closer point, which is chosen in a non-deterministic way (multiple threads).
        """
        k = 6
        max_iter = 50
        from pyemma.coordinates.clustering.tests.util import make_blobs
        data = make_blobs(n_samples=500, random_state=45, centers=k, cluster_std=0.5, shuffle=False)[0]
        repeat = True
        it = 0
        # since this can fail in like one of 100 runs, we repeat until success.
        while repeat and it < 3:
            for strat in ('uniform', 'kmeans++'):
                seed = random.randint(0, 2**32-1)
                cl_serial = cluster_kmeans(data, k=k, n_jobs=0, fixed_seed=seed, max_iter=max_iter, init_strategy=strat)
                cl_parallel = cluster_kmeans(data, k=k, n_jobs=2, fixed_seed=seed, max_iter=max_iter, init_strategy=strat)
                try:
                    np.testing.assert_allclose(cl_serial.clustercenters, cl_parallel.clustercenters, atol=1e-4)
                    repeat = False
                except AssertionError:
                    repeat = True
                    it += 1

    def test_negative_seed(self):
        """ ensure negative seeds converted to something positive"""
        km = cluster_kmeans(np.random.random((10, 3)), k=2, fixed_seed=-1)
        self.assertGreaterEqual(km.fixed_seed, 0)

    def test_seed_too_large(self):
        km = cluster_kmeans(np.random.random((10, 3)), k=2, fixed_seed=2**32)
        assert km.fixed_seed < 2**32

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
        initial_centers_equilibrium = np.array([0, 0, 0])
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
        np.testing.assert_equal(initial_centers_equilibrium.squeeze(), kmeans.clustercenters.squeeze(), err_msg=msg)

    def test_kmeans_converge_outlier_to_equilibrium_state(self):
        initial_centers_equilibrium = np.array([[2, 0, 0], [-2, 0, 0]])
        X = np.array([
            np.array([1, 1.5, 1], dtype=np.float32), np.array([1, 1, -1], dtype=np.float32),
            np.array([1, -1, -1], dtype=np.float32), np.array([-1, -1, -1], dtype=np.float32),
            np.array([-1, 1, 1], dtype=np.float32), np.array([-1, -1, 1], dtype=np.float32),
            np.array([-1, 1, -1], dtype=np.float32), np.array([1, -1, 1], dtype=np.float32)
        ])
        kmeans = cluster_kmeans(X, k=2, clustercenters=initial_centers_equilibrium, max_iter=500, n_jobs=1)

        cl = kmeans.clustercenters
        assert np.all(np.abs(cl) <= 1)

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
        kmeans = cluster_kmeans(np.random.rand(500, 3), 10, metric='minRMSD')
        kmeans.dtrajs

    def test_skip(self):
        cl = cluster_kmeans(np.random.rand(100, 3), skip=42)
        assert len(cl.dtrajs[0]) == 100 - 42

    def test_with_pg(self):
        from pyemma.util.contexts import settings
        with settings(show_progress_bars=True):
            cluster_kmeans(np.random.rand(100, 3))


class TestKmeansResume(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from pyemma.util.contexts import numpy_random_seed
        with numpy_random_seed(32):
            # three gaussians
            X = [np.random.randn(1000)-2.0,
                 np.random.randn(1000),
                 np.random.randn(1000)+2.0]
            cls.X = np.hstack(X)

    def test_resume(self):
        """ check that we can continue with the iteration by passing centers"""
        initial_centers = np.array([[20, 42, -29]]).T
        cl = cluster_kmeans(self.X, clustercenters=initial_centers,
                            max_iter=1, k=3, keep_data=True)

        resume_centers = cl.clustercenters
        cl.estimate(self.X, clustercenters=resume_centers, max_iter=50)
        new_centers = cl.clustercenters

        true = np.array([-2, 0, 2])
        d0 = true - resume_centers
        d1 = true - new_centers

        diff = np.linalg.norm(d0)
        diff_next = np.linalg.norm(d1)

        self.assertLess(diff_next, diff, 'resume_centers=%s, new_centers=%s' % (resume_centers, new_centers))

    @unittest.skip('not impled')
    def test_inefficient_args_log(self):
        from pyemma.util.testing_tools import MockLoggingHandler
        m = MockLoggingHandler()
        cl = cluster_kmeans(self.X, max_iter=1, keep_data=False)
        cl.logger.addHandler(m)
        cl.estimate(self.X, max_iter=1, clustercenters=cl.clustercenters)
        found = False
        for msg in m.messages['warning']:
            if 'inefficient' in msg:
                found = True
                break

        assert found

    def test_converged_memory_freed(self):
        k = 3
        initial_centers = np.atleast_2d(self.X[np.random.choice(1000, size=k)]).T

        cl = cluster_kmeans(self.X, clustercenters=initial_centers, k=k, max_iter=1, keep_data=True)

        while not cl.converged:
            cl.estimate(self.X, clustercenters=cl.clustercenters, max_iter=5)

        assert not hasattr(cl, '_in_memory_chunks')

    def test_non_converged_keep_memory(self):
        k = 3
        initial_centers = np.atleast_2d(self.X[np.random.choice(1000, size=k)]).T

        cl = cluster_kmeans(self.X, clustercenters=initial_centers, k=k, max_iter=1, keep_data=True)

        cl.estimate(self.X, clustercenters=cl.clustercenters, max_iter=1)
        assert not cl.converged
        assert hasattr(cl, '_in_memory_chunks')

    def test_syntetic_trivial(self):
        test_data = np.zeros((40000, 4))
        test_data[0:10000, :] = 30.0
        test_data[10000:20000, :] = 60.0
        test_data[20000:30000, :] = 90.0
        test_data[30000:, :] = 120.0

        expected = np.array([30.0]*4), np.array([60.]*4), np.array([90.]*4), np.array([120.]*4)
        cl = cluster_kmeans(test_data, k=4)
        found = [False]*4
        for center in cl.clustercenters:
            for i, e in enumerate(expected):
                if np.all( center == e):
                    found[i] = True

        assert np.all(found)

if __name__ == "__main__":
    unittest.main()
