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
Created on 23.01.2015

@author: marscher
'''
import mdtraj
import os
import tempfile
import unittest
from pyemma.coordinates import api
from pyemma.coordinates.data.feature_reader import FeatureReader
from pyemma.util.log import getLogger
import pkg_resources

import numpy as np
from pyemma.coordinates.api import feature_reader, discretizer, tica

log = getLogger('TestFeatureReader')


class TestFeatureReader(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        c = super(TestFeatureReader, cls).setUpClass()
        # create a fake trajectory which has 3 atoms and coordinates are just a range
        # over all frames.
        cls.trajfile = tempfile.mktemp('.xtc')
        cls.n_frames = 1000
        cls.xyz = np.random.random(
            cls.n_frames * 3 * 3).reshape((cls.n_frames, 3, 3))
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
        try:
            os.unlink(cls.trajfile)
        except EnvironmentError:
            pass

    def testIteratorAccess(self):
        reader = api.source(self.trajfile, top=self.topfile)

        frames = 0
        data = []
        for i, X in reader:
            frames += X.shape[0]
            data.append(X)

        # restore shape of input
        data = np.array(data).reshape(self.xyz.shape)

        self.assertEqual(frames, reader.trajectory_lengths()[0])
        self.assertTrue(np.allclose(data, self.xyz))

    def testIteratorAccess2(self):
        reader = FeatureReader([self.trajfile, self.trajfile], self.topfile)
        reader.chunksize = 100

        frames = 0
        data = []
        for i, X in reader:
            frames += X.shape[0]
            data.append(X)
        self.assertEqual(frames, reader.trajectory_lengths()[0] * 2)
        # restore shape of input
        data = np.array(
            data[0:reader.trajectory_lengths()[0] / reader.chunksize]).reshape(self.xyz.shape)

        self.assertTrue(np.allclose(data, self.xyz))

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
        fake_lagged = np.empty((xyz_s[0] - lag, xyz_s[1] * xyz_s[2]))
        fake_lagged = self.xyz.reshape((xyz_s[0], -1))[lag:]

        self.assertTrue(np.allclose(merged_lagged, fake_lagged))

        # restore shape of input
        data = np.array(data).reshape(self.xyz.shape)

        self.assertEqual(frames, reader.trajectory_lengths()[0])
        self.assertTrue(np.allclose(data, self.xyz))

    def test_with_pipeline_time_lagged(self):
        reader = feature_reader(self.trajfile, self.topfile)
        #reader.featurizer.distances([[0, 1], [0, 2]])
        t = tica(dim=2, lag=1)
        d = discretizer(reader, t)
        d.parametrize()

    def test_in_memory(self):
        reader = api.source(self.trajfile, top=self.topfile)
        out1 = reader.get_output()
        # now map stuff to memory
        reader.in_memory = True

        print len(reader._Y)
        print reader._Y[0].shape

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
        assert reader._parametrized
        reader.parametrize(stride=2)

        reader2 = api.source(self.trajfile, top=self.topfile)
        out = reader2.get_output(stride=2)

        np.testing.assert_equal(reader._Y[0], out[0])

    def test_in_memory_switch_stride_dim(self):
        reader = api.source(self.trajfile, top=self.topfile)
        reader.chunksize = 360
        reader.in_memory = True

        # now get output with different strides
        strides = [1, 2, 3, 4, 5, 10, 20]
        for s in strides:
            if reader.chunksize % s != 0:
                continue
            out = reader.get_output(stride=s)
            shape = (reader.trajectory_length(0, stride=s), reader.dimension())
            self.assertEqual(out[0].shape, shape, "not equal for stride=%i" % s)

    def test_lagged_stridden_access(self):
        reader = api.source(self.trajfile, top=self.topfile)
        reader.chunksize = 210
        strides = [2, 3, 5, 7, 15]
        lags = [1, 3, 7, 10, 30]
        for stride in strides:
            for lag in lags:
                chunks = []
                for _, _, Y in reader.iterator(stride, lag):
                    chunks.append(Y)
                chunks = np.vstack(chunks)
                np.testing.assert_almost_equal(chunks, self.xyz.reshape(-1, 9)[lag::stride])

if __name__ == "__main__":
    unittest.main()
