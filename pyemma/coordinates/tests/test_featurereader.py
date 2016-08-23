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
Created on 23.01.2015

@author: marscher
'''

from __future__ import absolute_import

import glob
import tempfile
import unittest

from pyemma.coordinates import api
from pyemma.coordinates.api import discretizer, tica, source
from pyemma.coordinates.data.data_in_memory import DataInMemoryIterator
from pyemma.coordinates.data.feature_reader import FeatureReader
from logging import getLogger
import mdtraj
import pkg_resources

from six.moves import range
import numpy as np

from pyemma.coordinates.tests.util import create_traj

log = getLogger('pyemma.' + 'TestFeatureReader')


class TestFeatureReader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # create a fake trajectory which has 3 atoms and coordinates are just a range
        # over all frames.
        cls.tmpdir = tempfile.mkdtemp('test_feature_reader')

        cls.topfile = pkg_resources.resource_filename(__name__, 'data/test.pdb')
        cls.trajfile, cls.xyz, cls.n_frames = create_traj(cls.topfile, dir=cls.tmpdir)
        cls.trajfile2, cls.xyz2, cls.n_frames2 = create_traj(cls.topfile, dir=cls.tmpdir)

        return cls

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def testIteratorAccess(self):
        reader = api.source(self.trajfile, top=self.topfile)
        assert isinstance(reader, FeatureReader)

        frames = 0
        data = []
        for i, X in reader:
            assert isinstance(X, np.ndarray)
            frames += X.shape[0]
            data.append(X)

        self.assertEqual(frames, reader.trajectory_lengths()[0])
        data = np.vstack(data)
        # restore shape of input
        data.reshape(self.xyz.shape)

        self.assertTrue(np.allclose(data, self.xyz.reshape(-1, 9)))

    def testIteratorAccess2(self):
        reader = FeatureReader([self.trajfile, self.trajfile2], self.topfile)
        reader.chunksize = 100

        data = {itraj: [] for itraj in range(reader.number_of_trajectories())}

        for i, X in reader:
            data[i].append(X)

        # restore shape of input
        data[0] = np.vstack(data[0]).reshape(-1, 9)
        data[1] = np.vstack(data[1]).reshape(-1, 9)

        np.testing.assert_equal(data[0], self.xyz.reshape(-1, 9))
        np.testing.assert_equal(data[1], self.xyz2.reshape(-1, 9))

    def test_skip(self):
        for skip in [0, 3, 13]:
            r1 = FeatureReader(self.trajfile, self.topfile)
            out_with_skip = r1.get_output(skip=skip)[0]
            r2 = FeatureReader(self.trajfile, self.topfile)
            out = r2.get_output()[0]
            np.testing.assert_almost_equal(out_with_skip, out[skip::],
                                           err_msg="The first %s rows were skipped, but that did not "
                                                   "match the rows with skip=0 and sliced by [%s::]" % (skip, skip))

    def test_skip_input_list(self):
        for skip in [0, 3, 13]:
            r1 = FeatureReader([self.trajfile, self.trajfile2], self.topfile)
            out_with_skip = r1.get_output(skip=skip)
            r2 = FeatureReader([self.trajfile, self.trajfile2], self.topfile)
            out = r2.get_output()
            np.testing.assert_almost_equal(out_with_skip[0], out[0][skip::],
                                           err_msg="The first %s rows of the first file were skipped, but that did not "
                                                   "match the rows with skip=0 and sliced by [%s::]" % (skip, skip))
            np.testing.assert_almost_equal(out_with_skip[1], out[1][skip::],
                                           err_msg="The first %s rows of the second file were skipped, but that did not"
                                                   " match the rows with skip=0 and sliced by [%s::]" % (skip, skip))

    def testTimeLaggedIterator(self):
        lag = 10
        reader = FeatureReader(self.trajfile, self.topfile)
        frames = 0
        data = []
        lagged = []
        for _, X, Y in reader.iterator(lag=lag):
            frames += X.shape[0]
            data.append(X)
            lagged.append(Y)

        assert len(data) == len(lagged)
        # .reshape(self.xyz.shape)
        merged_lagged = np.concatenate(lagged, axis=0)

        # reproduce outcome
        xyz_s = self.xyz.shape
        fake_lagged = self.xyz.reshape((xyz_s[0], -1))[lag:]

        self.assertTrue(np.allclose(merged_lagged, fake_lagged))

        # restore shape of input
        data = np.vstack(data).reshape(self.xyz[lag:].shape)

        self.assertEqual(frames, reader.trajectory_lengths()[0] - lag)
        self.assertTrue(np.allclose(data, self.xyz[:len(self.xyz) - lag]))

    def test_cols(self):
        reader = api.source(self.trajfile, top=self.topfile)
        # select first and second atom?
        cols = np.array((0, 2))
        ref = mdtraj.load(self.trajfile, top=self.topfile).xyz
        s = ref.shape
        new_shape = (s[0], s[1] * s[2])
        ref = ref.reshape(new_shape)
        ref = ref[:, cols]

        it = reader.iterator(chunk=0, return_trajindex=False, cols=cols)
        with it:
            for x in it:
                np.testing.assert_equal(x, ref)

    def test_cols_with_features(self):
        trajs = glob.glob(pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/bpti_mini.xtc'))
        top = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/bpti_ca.pdb')
        reader = api.source(trajs, top=top)
        feat = reader.featurizer
        inds = feat.pairs(feat.select('name CA'))
        reader.featurizer.add_distances(inds)
        # select first and second atom?
        cols = np.array((0, 2))
        ref = mdtraj.load(trajs, top=top)
        ref = mdtraj.compute_distances(ref, inds)
        ref = ref[:, cols]

        it = reader.iterator(chunk=0, return_trajindex=False, cols=cols)
        with it:
            for x in it:
                np.testing.assert_equal(x, ref)

    def test_with_pipeline_time_lagged(self):
        reader = api.source(self.trajfile, top=self.topfile)
        assert isinstance(reader, FeatureReader)

        t = tica(dim=2, lag=1)
        d = discretizer(reader, t, chunksize=10)
        d.parametrize()

    def test_in_memory(self):
        reader = api.source(self.trajfile, top=self.topfile)
        out1 = reader.get_output()
        # now map stuff to memory
        reader.in_memory = True

        reader2 = api.source(self.trajfile, top=self.topfile)
        out = reader2.get_output()

        assert len(out) == len(reader._Y) == 1
        np.testing.assert_equal(out1, out)
        np.testing.assert_equal(reader._Y[0], out[0])
        np.testing.assert_equal(reader.get_output(), out)

        # reset in_memory and check output gets deleted
        reader.in_memory = False
        assert reader._Y is None

    def test_in_memory_with_stride(self):
        # map "results" to memory
        reader = api.source(self.trajfile, top=self.topfile)
        reader.in_memory = True
        mem_it = reader.iterator(stride=2, chunk=0, return_trajindex=False)
        assert isinstance(mem_it, DataInMemoryIterator)
        mem_data = [X for X in mem_it]

        reader2 = api.source(self.trajfile, top=self.topfile)
        out = reader2.get_output(stride=2)

        np.testing.assert_equal(mem_data[0], out[0])

    def test_in_memory_switch_stride_dim(self):
        reader = api.source(self.trajfile, top=self.topfile)
        reader.chunksize = 100
        reader.in_memory = True

        # now get output with different strides
        strides = [1, 2, 3, 4, 5]
        for s in strides:
            out = reader.get_output(stride=s)
            shape = (reader.trajectory_length(0, stride=s), reader.dimension())
            self.assertEqual(out[0].shape, shape, "not equal for stride=%i" % s)

    def test_lagged_stridden_access(self):
        reader = api.source([self.trajfile, self.trajfile2], top=self.topfile)
        reader.chunksize = 210
        strides = [2, 3, 5, 7, 15]
        lags = [1, 3, 7, 10, 30]
        err_msg = "not equal for stride=%i, lag=%i"
        for stride in strides:
            for lag in lags:
                chunks = {itraj: [] for itraj in range(reader.number_of_trajectories())}
                for itraj, _, Y in reader.iterator(stride=stride, lag=lag):
                    chunks[itraj].append(Y)
                chunks[0] = np.vstack(chunks[0])
                np.testing.assert_almost_equal(
                    chunks[0], self.xyz.reshape(-1, 9)[lag::stride], err_msg=err_msg % (stride, lag))

                chunks[1] = np.vstack(chunks[1])
                np.testing.assert_almost_equal(
                    chunks[1], self.xyz2.reshape(-1, 9)[lag::stride], err_msg=err_msg % (stride, lag))

    def test_lagged_access_small_files(self):
        """ itraj 0 should be skipped, since it is too short."""
        top = self.topfile
        trajs = [create_traj(top=top, length=10, format='.xtc', dir=self.tmpdir)[0],
                 create_traj(top=top, length=20, format='.xtc', dir=self.tmpdir)[0]]

        reader = source(trajs, top=top)
        it = reader.iterator(lag=11, chunk=0)
        res = {}
        with it:
            for itraj, x, y in it:
                res[itraj] = (x.shape, y.shape)

        self.assertNotIn(0, res)
        self.assertIn(1, res)

if __name__ == "__main__":
    unittest.main()
