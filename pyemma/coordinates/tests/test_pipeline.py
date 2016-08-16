
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



from __future__ import print_function

from __future__ import absolute_import
import unittest
import os

import numpy as np

from pyemma.coordinates.data import DataInMemory
from pyemma.coordinates.data import MDFeaturizer
from pyemma.coordinates import api
import msmtools.generation as msmgen
import tempfile
from six.moves import range
import pkg_resources
from pyemma.util.files import TemporaryDirectory
import pyemma.coordinates as coor

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
        cls.pdb_file = os.path.join(path, 'bpti_ca.pdb')
        cls.feat = MDFeaturizer(cls.pdb_file)
        cls.feat.add_all()
        cls.traj_files = [
            os.path.join(path, 'bpti_001-033.xtc'),
            os.path.join(path, 'bpti_067-100.xtc')
        ]

        # generate HMM with two gaussians
        p = np.array([[0.99, 0.01], [0.01, 0.99]])
        t = 10000
        means = [np.array([-1, 1]), np.array([1, -1])]
        widths = [np.array([0.3, 2]), np.array([0.3, 2])]
        # continuous trajectory
        x = np.zeros((t, 2))
        # hidden trajectory
        dtraj = msmgen.generate_traj(p, t)
        for t in range(t):
            s = dtraj[t]
            x[t, 0] = widths[s][0] * np.random.randn() + means[s][0]
            x[t, 1] = widths[s][1] * np.random.randn() + means[s][1]
        cls.generated_data = x
        cls.generated_lag = 10

    def test_is_parametrized(self):
        # construct pipeline with all possible transformers
        p = api.pipeline(
            [
                api.source(self.traj_files, top=self.pdb_file),
                api.tica(),
                api.pca(),
                api.cluster_kmeans(k=50),
                api.cluster_regspace(dmin=50),
                api.cluster_uniform_time(k=20)
            ], run=False
        )
        self.assertFalse(p._is_estimated(), "If run=false, the pipeline should not be parametrized.")
        p.parametrize()
        self.assertTrue(p._is_estimated(), "If parametrized was called, the pipeline should be parametrized.")

    def test_notify_changes_mixin(self):
        X_t = np.random.random((30,30))
        source = coor.source(np.array(X_t))

        t1 = coor.tica(source)
        from pyemma.coordinates.transform import TICA
        t2 = TICA(lag=10)
        assert len(t1._stream_children) == 0
        t2.data_producer = t1
        assert t1._stream_children[0] == t2

    def test_np_reader_in_pipeline(self):
        with TemporaryDirectory() as td:
            file_name = os.path.join(td, "test.npy")
            data = np.random.random((100, 3))
            np.save(file_name, data)
            reader = api.source(file_name)
            p = api.pipeline(reader, run=False, stride=2, chunksize=5)
            p.parametrize()

    def test_add_element(self):
        # start with empty pipeline without auto-parametrization
        p = api.pipeline([], run=False)
        # add some reader
        reader = api.source(self.traj_files, top=self.pdb_file)
        p.add_element(reader)
        p.parametrize()

        # get the result immediately
        out1 = reader.get_output()

        # add some kmeans
        kmeans = api.cluster_kmeans(k=15)
        p.add_element(kmeans)
        p.parametrize()
        # get the result immediately
        kmeans1 = kmeans.get_output()

        # get reader output again
        out2 = reader.get_output()
        p.add_element(api.cluster_kmeans(k=2))
        p.parametrize()

        # get kmeans output again
        kmeans2 = kmeans.get_output()
        # check if add_element changes the intermediate results
        np.testing.assert_array_equal(out1[0], out2[0])
        np.testing.assert_array_equal(out1[1], out2[1])
        np.testing.assert_array_equal(kmeans1[0], kmeans2[0])
        np.testing.assert_array_equal(kmeans1[1], kmeans2[1])

    def test_set_element(self):
        reader = api.source(self.traj_files, top=self.pdb_file)
        pca = api.pca()
        p = api.pipeline([reader, pca])
        self.assertTrue(p._is_estimated())
        pca_out = pca.get_output()
        tica = api.tica(lag=self.generated_lag)
        # replace pca with tica
        p.set_element(1, tica)
        self.assertFalse(p._is_estimated(), "After replacing an element, the pipeline should not be parametrized.")
        p.parametrize()
        tica_out = tica.get_output()
        # check if replacement actually happened
        self.assertFalse(np.array_equal(pca_out[0], tica_out[0]),
                         "The output should not be the same when the method got replaced.")

    @unittest.skip("Known to be broken")
    def test_replace_data_source(self):
        reader_xtc = api.source(self.traj_files, top=self.pdb_file)
        reader_gen = DataInMemory(data=self.generated_data)

        kmeans = api.cluster_kmeans(k=10)
        assert hasattr(kmeans, '_chunks')
        p = api.pipeline([reader_xtc, kmeans])
        out1 = kmeans.get_output()
        # replace source
        print(reader_gen)
        p.set_element(0, reader_gen)
        assert hasattr(kmeans, '_chunks')
        p.parametrize()
        out2 = kmeans.get_output()
        self.assertFalse(np.array_equal(out1, out2), "Data source changed, so should the resulting clusters.")

    def test_discretizer(self):
        reader_gen = DataInMemory(data=self.generated_data)
        # check if exception safe
        api.discretizer(reader_gen)._chain[-1].get_output()
        api.discretizer(reader_gen, transform=api.tica())._chain[-1].get_output()
        api.discretizer(reader_gen, cluster=api.cluster_uniform_time())._chain[-1].get_output()
        api.discretizer(reader_gen, transform=api.pca(), cluster=api.cluster_regspace(dmin=10))._chain[-1].get_output()

    def test_no_cluster(self):
        reader_xtc = api.source(self.traj_files, top=self.pdb_file)
        # only reader
        api.pipeline(reader_xtc)
        reader_xtc.get_output()
        # reader + pca / tica
        tica = api.tica()
        pca = api.pca()
        api.pipeline([reader_xtc, tica])._chain[-1].get_output()
        api.pipeline([reader_xtc, pca])._chain[-1].get_output()

    def test_no_transform(self):
        reader_xtc = api.source(self.traj_files, top=self.pdb_file)
        api.pipeline([reader_xtc, api.cluster_kmeans(k=10)])._chain[-1].get_output()
        api.pipeline([reader_xtc, api.cluster_regspace(dmin=10)])._chain[-1].get_output()
        api.pipeline([reader_xtc, api.cluster_uniform_time()])._chain[-1].get_output()

    def test_chunksize(self):
        reader_xtc = api.source(self.traj_files, top=self.pdb_file)
        chunksize = 1001
        chain = [reader_xtc, api.tica(), api.cluster_mini_batch_kmeans()]
        p = api.pipeline(chain, chunksize=chunksize)
        assert p.chunksize == chunksize
        for e in p._chain:
            assert e.chunksize == chunksize

if __name__ == "__main__":
    unittest.main()