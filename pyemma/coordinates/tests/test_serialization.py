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

import os
import tempfile
import unittest

import numpy as np

import pyemma
import pyemma.coordinates as coor


class TestSerializationCoordinates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = np.random.random((100, 3))

    def setUp(self):
        self.fn = tempfile.mktemp()

    def tearDown(self):
        try:
            os.unlink(self.fn)
        except:
            pass

    def compare(self, obj, params):
        """

        Parameters
        ----------
        obj
        params

        Returns
        -------

        """
        fn = self.fn
        obj.save(fn)
        restored = pyemma.load(fn)

        for k, v in params.iteritems():
            actual = getattr(restored, k)
            expected = getattr(obj, k)
            if isinstance(actual, np.ndarray):
                np.testing.assert_equal(actual, expected)
            else:
                self.assertEqual(actual, expected)

    def test_clustering_kmeans(self):
        params = {'k': 10, 'init_strategy': 'uniform', 'max_iter': 42,
                  'metric': 'minRMSD', 'stride': 1}
        cl = coor.cluster_kmeans(**params)
        params['n_clusters'] = params['k']
        del params['k']

        self.compare(cl, params)

    def test_clustering_regspace(self):
        params = {'dmin': 0.1, 'max_centers': 100, 'metric': 'minRMSD', 'n_jobs': 4, 'stride': 2}

        cl = pyemma.coordinates.cluster_regspace(**params)
        self.compare(cl, params)

    def test_clustering_uniform_time(self):
        params = {'k': 3, 'metric': 'minRMSD', 'n_jobs': 4}
        cl = pyemma.coordinates.cluster_uniform_time(**params)
        params['n_clusters'] = params['k']
        del params['k']

        self.compare(cl, params)

    def test_clustering_minibatch_kmeans(self):
        params = {'k': 10, 'init_strategy': 'uniform', 'max_iter': 42,
                  'metric': 'minRMSD'}
        cl = coor.cluster_mini_batch_kmeans(**params)
        params['n_clusters'] = params['k']
        del params['k']

        self.compare(cl, params)

    def test_tica(self):
        params = {'lag': 10, 'dim': 3, 'kinetic_map': True,
                  'stride': 2}
        cl = pyemma.coordinates.tica(**params)
        self.compare(cl, params)

    def test_tica_estimated(self):
        params = {'lag': 10, 'dim': 3, 'kinetic_map': True,
                  'mean': None, 'remove_mean': True, 'stride': 2}
        t = pyemma.coordinates.tica(data=self.data, **params)

        assert t._estimated
        params['cov'] = t.cov
        params['cov_tau'] = t.cov_tau
        params['eigenvalues'] = t.eigenvalues
        params['eigenvectors'] = t.eigenvectors

        self.compare(t, params)

    def test_pca(self):
        params = {'dim': 3,
                  'mean': None, 'stride': 2}
        p = pyemma.coordinates.pca(**params)

        self.compare(p, params)

    def test_pca_estimated(self):
        params = {'dim': 3,
                  'mean': None, 'stride': 2}

        t = pyemma.coordinates.pca(data=self.data, **params)
        assert t._estimated
        params['cov'] = t.cov
        params['mean'] = t.mean
        params['eigenvalues'] = t.eigenvalues
        params['eigenvectors'] = t.eigenvectors

        self.compare(t, params)

    @unittest.skip("not yet impled.")
    def test_featurizer(self):
        f = pyemma.coordinates.featurizer()
