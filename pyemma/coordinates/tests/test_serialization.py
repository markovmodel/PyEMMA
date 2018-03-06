# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import pkg_resources
import six

import pyemma
import pyemma.coordinates as coor
from pyemma.coordinates.data.numpy_filereader import NumPyFileReader

@unittest.skipIf(six.PY2, 'only py3')
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

        for k, v in params.items():
            actual = getattr(restored, k)
            expected = getattr(obj, k)
            if isinstance(actual, np.ndarray):
                np.testing.assert_equal(actual, expected)
            elif isinstance(actual, list):
                self.assertListEqual(actual, expected)
            else:
                self.assertEqual(actual, expected)
        # return the restored obj for further evaluation
        return restored

    def test_clustering_kmeans(self):
        params = {'k': 10, 'init_strategy': 'uniform', 'max_iter': 42,
                  'metric': 'minRMSD', 'stride': 1}
        cl = coor.cluster_kmeans([np.random.random((100, 3))],**params)
        params['n_clusters'] = params['k']
        params['clustercenters'] = cl.clustercenters  # this is a model param, so it should contained in the output
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
                  'stride': 2}
        t = pyemma.coordinates.tica(data=self.data, **params)

        assert t._estimated
        params['cov'] = t.cov
        params['cov_tau'] = t.cov_tau
        params['eigenvalues'] = t.eigenvalues
        params['eigenvectors'] = t.eigenvectors

        self.compare(t, params)

    def test_pca(self):
        params = {'dim': 3,
                  'stride': 2}
        p = pyemma.coordinates.pca(**params)

        self.compare(p, params)

    def test_pca_estimated(self):
        params = {'dim': 3,
                  'mean': None, 'stride': 2}

        t = pyemma.coordinates.pca(data=self.data, **params)
        assert t._estimated
        #params['cov'] = t.cov
        params['mean'] = t.mean
        #params['eigenvalues'] = t.eigenvalues
        params['eigenvectors'] = t.eigenvectors

        self.compare(t, params)

    def test_save_chain(self):
        """ ensure a chain is correctly saved/restored"""
        from pyemma.datasets import get_bpti_test_data

        reader = pyemma.coordinates.source(get_bpti_test_data()['trajs'], top=get_bpti_test_data()['top'])
        tica = pyemma.coordinates.tica(reader)
        cluster = pyemma.coordinates.cluster_uniform_time(tica, 10)

        cluster.save(self.fn, save_streaming_chain=True)
        restored = pyemma.load(self.fn)
        self.assertIsInstance(restored, type(cluster))
        self.assertIsInstance(restored.data_producer, type(tica))
        self.assertIsInstance(restored.data_producer.data_producer, type(reader))
        cluster.save(self.fn, overwrite=True, save_streaming_chain=False)
        restored = pyemma.load(self.fn)
        assert restored.data_producer is None

    def test_featurizer_empty(self):
        from pyemma.datasets import get_bpti_test_data
        top = get_bpti_test_data()['top']
        f = pyemma.coordinates.featurizer(top)
        params = {}
        params['topologyfile'] = top

        self.compare(f, params)

    def test_featurizer(self):
        from pyemma.datasets import get_bpti_test_data
        top = get_bpti_test_data()['top']
        f = pyemma.coordinates.featurizer(top)
        f.add_distances_ca()
        params = {}
        params['topologyfile'] = top
        params['active_features'] = f.active_features
        self.maxDiff = None
        self.compare(f, params)

    def test_feature_reader(self):
        from pyemma.datasets import get_bpti_test_data
        top = get_bpti_test_data()['top']
        trajs = get_bpti_test_data()['trajs']
        r = pyemma.coordinates.source(trajs, top=top)
        r.featurizer.add_distances_ca()

        params = {'filenames': trajs, 'ndim': r.ndim, 'topfile': r.topfile}
        restored = self.compare(r, params=params)
        assert hasattr(restored, '_is_reader')
        assert restored._is_reader
        self.assertEqual(restored.featurizer.active_features, r.featurizer.active_features)

    def test_numpy_reader(self):
        arr = np.random.random(10)
        from pyemma.util.files import TemporaryDirectory
        with TemporaryDirectory() as d:
            files = [os.path.join(d, '1.npy'), os.path.join(d, '2.npy')]
            np.save(files[0], arr)
            np.save(files[1], arr)
            params = {'filenames': files, 'chunksize': 23}
            r = NumPyFileReader(**params)
            self.compare(r, params)

    def test_csv_reader(self):
        arr = np.random.random(10).reshape(-1, 2)
        from pyemma.util.files import TemporaryDirectory
        delimiter = ' '
        with TemporaryDirectory() as d:
            files = [os.path.join(d, '1.csv'), os.path.join(d, '2.csv')]
            np.savetxt(files[0], arr, delimiter=delimiter)
            np.savetxt(files[1], arr, delimiter=delimiter)
            params = {'filenames': files, 'chunksize': 23}
            from pyemma.coordinates.data import PyCSVReader
            # sniffing the delimiter does not aid in the 1-column case:
            # https://bugs.python.org/issue2078
            # but also specifying it does not help...
            r = PyCSVReader(delimiter=delimiter, **params)
            self.compare(r, params)

    def test_fragmented_reader(self):
        from pyemma.coordinates.tests.util import create_traj
        from pyemma.util.files import TemporaryDirectory

        top_file = pkg_resources.resource_filename(__name__, 'data/test.pdb')
        trajfiles = []

        with TemporaryDirectory() as d:
            for _ in range(3):
                f, _, _ = create_traj(top_file, dir=d)
                trajfiles.append(f)
            # three trajectories: one consisting of all three, one consisting of the first,
            # one consisting of the first and the last
            frag_trajs = [trajfiles, [trajfiles[0]], [trajfiles[0], trajfiles[2]]]
            chunksize = 232
            source = coor.source(frag_trajs, top=top_file, chunksize=chunksize)
            params = {'chunksize': chunksize, 'ndim': source.ndim, '_trajectories': trajfiles}
            restored = self.compare(source, params)

            np.testing.assert_equal(source.get_output(), restored.get_output())

    def test_h5_reader(self):
        h5_file = pkg_resources.resource_filename(__name__, 'data/bpti_mini.h5')
        params = dict(selection='/coordinates')
        source = coor.source(h5_file, **params)
        restored = self.compare(source, params)
        np.testing.assert_equal(source.get_output(), restored.get_output())


if __name__ == '__main__':
    unittest.main()
