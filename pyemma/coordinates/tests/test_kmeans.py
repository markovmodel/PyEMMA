
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on 28.01.2015

@author: marscher
'''
import unittest
import tempfile
import os
import numpy as np
from pyemma.coordinates.api import cluster_kmeans
import shutil


class TestKmeans(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestKmeans, cls).setUpClass()
        cls.dtraj_dir = tempfile.mkdtemp()

    def setUp(self):
        self.k = 5
        self.dim = 100
        self.data = [np.random.random((30, self.dim)),
                     np.random.random((37, self.dim))]
        self.kmeans = cluster_kmeans(data=self.data, k=self.k, max_iter=100)

    def tearDown(self):
        shutil.rmtree(self.dtraj_dir, ignore_errors=True)

    def testDtrajType(self):
        assert self.kmeans.dtrajs[0].dtype == self.kmeans.output_type()

    def test_3gaussian_1d_singletraj(self):
        # generate 1D data from three gaussians
        X = [np.random.randn(100)-2.0,
             np.random.randn(100),
             np.random.randn(100)+2.0]
        X = np.hstack(X)
        kmeans = cluster_kmeans(X, k=10)
        cc = kmeans.clustercenters
        assert(np.any(cc < 1.0))
        assert(np.any((cc > -1.0) * (cc < 1.0)))
        assert(np.any(cc > -1.0))

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

    def testSaveDtrajs(self):
        prefix = "test"
        extension = ".dtraj"
        outdir = self.dtraj_dir
        self.kmeans.save_dtrajs(trajfiles=None, prefix=prefix,
                                output_dir=outdir, extension=extension)

        names = ["%s_%i%s" % (prefix, i, extension)
                 for i in xrange(self.kmeans.data_producer.number_of_trajectories())]
        names = [os.path.join(outdir, n) for n in names]

        # check files with given patterns are there
        for f in names:
            os.stat(f)

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

if __name__ == "__main__":
    unittest.main()