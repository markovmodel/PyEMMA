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
from unittest import TestCase

import mdtraj
import numpy as np
import pkg_resources
from six.moves import range

import pyemma.coordinates.api as coor
from pyemma.coordinates.data import DataInMemory, FeatureReader
from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader
from pyemma.coordinates.tests.util import create_traj, get_top
from pyemma.util.files import TemporaryDirectory


def _test_ra_with_format(format, stride):
    from pyemma.coordinates.tests.test_featurereader import create_traj

    topfile = pkg_resources.resource_filename(__name__, 'data/test.pdb')
    trajfiles = []
    for _ in range(3):
        f, _, _ = create_traj(topfile, format=format)
        trajfiles.append(f)
    try:
        source = coor.source(trajfiles, top=topfile)
        source.chunksize = 2

        out = source.get_output(stride=stride)
        keys = np.unique(stride[:, 0])
        for i, coords in enumerate(out):
            if i in keys:
                traj = mdtraj.load(trajfiles[i], top=topfile)
                np.testing.assert_equal(coords,
                                        traj.xyz[
                                            np.array(stride[stride[:, 0] == i][:, 1])
                                        ].reshape(-1, 9))
    finally:
        for t in trajfiles:
            try:
                os.unlink(t)
            except EnvironmentError:
                pass

class TestRandomAccessStride(TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp('test_random_access')
        self.dim = 5
        self.data = [np.random.random((100, self.dim)).astype(np.float32),
                     np.random.random((20, self.dim)).astype(np.float32),
                     np.random.random((20, self.dim)).astype(np.float32)]
        self.stride = np.asarray([
            [0, 1], [0, 3], [0, 3], [0, 5], [0, 6], [0, 7],
            [2, 1], [2, 1]
        ])
        self.stride2 = np.asarray([[2, 0]])
        self.topfile = pkg_resources.resource_filename(__name__, 'data/test.pdb')
        trajfile1, xyz1, n_frames1 = create_traj(self.topfile, dir=self.tmpdir, format=".binpos", length=100)
        trajfile2, xyz2, n_frames2 = create_traj(self.topfile, dir=self.tmpdir, format=".binpos", length=20)
        trajfile3, xyz3, n_frames3 = create_traj(self.topfile, dir=self.tmpdir, format=".binpos", length=20)
        self.data_feature_reader = [trajfile1, trajfile2, trajfile3]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _get_reader_instance(self, instance_number):
        if instance_number == 0:
            return DataInMemory(self.data)
        elif instance_number == 1:
            return FeatureReader(self.data_feature_reader, topologyfile=self.topfile)

    def test_is_random_accessible(self):
        dim = DataInMemory(self.data)
        frag = FragmentedTrajectoryReader([[self.data]])
        assert dim.is_random_accessible is True
        assert frag.is_random_accessible is False

    def test_slice_random_access(self):
        dim = DataInMemory(self.data)

        all_data = dim.ra_itraj_cuboid[:, :, :]
        # the remaining 80 frames of the first trajectory should be truncated
        np.testing.assert_equal(all_data.shape, (3, 20, self.dim))
        # should coincide with original data
        np.testing.assert_equal(all_data, np.array((self.data[0][:20], self.data[1], self.data[2])))
        # we should be able to select the 1st trajectory
        np.testing.assert_equal(dim.ra_itraj_cuboid[0], np.array([self.data[0]]))
        # select only dimensions 1:3 of 2nd trajectory with every 2nd frame
        np.testing.assert_equal(dim.ra_itraj_cuboid[1, ::2, 1:3], np.array([self.data[1][::2, 1:3]]))
        # select only last dimension of 1st trajectory every 17th frame
        np.testing.assert_equal(dim.ra_itraj_cuboid[0, ::17, -1], np.array([np.array([self.data[0][::17, -1]]).T]))

    def test_transfomer_random_access(self):
        for in_memory in [True, False]:
            for r in range(0, 2):
                dim = self._get_reader_instance(r)

                tica = coor.tica(dim, dim=3)
                tica.in_memory = in_memory
                out = tica.get_output()

                # linear random access
                np.testing.assert_array_equal(np.squeeze(tica.ra_linear[0:2, 0]), out[0][0:2, 0])
                # linear itraj random access
                np.testing.assert_array_equal(np.squeeze(tica.ra_itraj_linear[0, :12, 0]), out[0][:12, 0])
                # jagged random access
                jagged = tica.ra_itraj_jagged[:, ::-3, 0]
                for i, X in enumerate(jagged):
                    np.testing.assert_array_equal(X, out[i][::-3, 0])
                # cuboid random access
                cube = tica.ra_itraj_cuboid[:, 0, 0]
                for i in range(3):
                    np.testing.assert_array_equal(cube[i], out[i][0, 0])

    def test_transformer_random_access_in_memory(self):
        feature_reader = self._get_reader_instance(1)
        tica = coor.tica(feature_reader)
        # everything normal
        assert tica.is_random_accessible
        from pyemma.coordinates.transform.transformer import StreamingTransformerRandomAccessStrategy
        assert isinstance(tica._ra_jagged, StreamingTransformerRandomAccessStrategy)

        # set to memory
        tica.in_memory = True
        assert tica.is_random_accessible
        from pyemma.coordinates.data.data_in_memory import DataInMemoryJaggedRandomAccessStrategy
        assert isinstance(tica._ra_jagged, DataInMemoryJaggedRandomAccessStrategy)

        # not in memory anymore, expect to fall back
        tica.in_memory = False
        assert tica.is_random_accessible
        from pyemma.coordinates.transform.transformer import StreamingTransformerRandomAccessStrategy
        assert isinstance(tica._ra_jagged, StreamingTransformerRandomAccessStrategy)

        # remove data source
        tica.data_producer = None
        assert not tica.is_random_accessible
        assert tica._ra_jagged is None

    def test_linear_random_access_with_mixed_trajs(self):
        for r in range(0, 2):
            dim = self._get_reader_instance(r)
            Y = dim.get_output()

            X = dim.ra_linear[np.array([1, 115, 2, 139, 0])]
            np.testing.assert_equal(X[0, :], Y[0][1, :])
            np.testing.assert_equal(X[1, :], Y[1][15, :])
            np.testing.assert_equal(X[2, :], Y[0][2, :])
            np.testing.assert_equal(X[3, :], Y[2][19, :])
            np.testing.assert_equal(X[4, :], Y[0][0, :])

    def test_cuboid_random_access_with_mixed_trajs(self):
        for r in range(0, 2):
            dim = self._get_reader_instance(r)
            output = dim.get_output()

            # take two times the first trajectory, one time the third with
            # two times the third frame and one time the second, each
            trajs = np.array([0, 2, 0])
            frames = np.array([2, 2, 1])
            X = dim.ra_itraj_cuboid[trajs, frames]
            np.testing.assert_equal(X[0], output[0][frames])
            np.testing.assert_equal(X[1], output[2][frames])
            np.testing.assert_equal(X[2], output[0][frames])

    def test_linear_itraj_random_access_with_mixed_trajs(self):
        for r in range(0, 2):
            dim = self._get_reader_instance(r)
            Y = dim.get_output()

            itrajs = np.array([2, 2, 0])
            frames = np.array([3, 23, 42])
            X = dim.ra_itraj_linear[itrajs, frames]

            np.testing.assert_equal(X[0], Y[2][3, :])
            np.testing.assert_equal(X[1], Y[2][3, :])
            np.testing.assert_equal(X[2], Y[0][2, :])

    def test_jagged_random_access_with_mixed_trajs(self):
        for r in range(0, 2):
            dim = self._get_reader_instance(r)
            Y = dim.get_output()

            itrajs = np.array([2, 2, 0])
            X = dim.ra_itraj_jagged[itrajs, ::-3]  #
            np.testing.assert_array_almost_equal(X[0], Y[2][::-3])
            np.testing.assert_array_almost_equal(X[1], Y[2][::-3])
            np.testing.assert_array_almost_equal(X[2], Y[0][::-3])

    def test_slice_random_access_linear(self):
        dim = DataInMemory(self.data)

        all_data = dim.ra_linear[:, :]
        # all data should be all data concatenated
        np.testing.assert_equal(all_data, np.concatenate(self.data))
        # select first 5 frames
        np.testing.assert_equal(dim.ra_linear[:5], self.data[0][:5])
        # select only dimensions 1:3 of every 2nd frame
        np.testing.assert_equal(dim.ra_linear[::2, 1:3], np.concatenate(self.data)[::2, 1:3])

    def test_slice_random_access_linear_itraj(self):
        dim = DataInMemory(self.data)

        all_data = dim.ra_itraj_linear[:, :, :]
        # all data should be all data concatenated
        np.testing.assert_equal(all_data, np.concatenate(self.data))

        # if requested 130 frames, this should yield the first two trajectories and half of the third
        np.testing.assert_equal(dim.ra_itraj_linear[:, :130], np.concatenate(self.data)[:130])
        # now request first 30 frames of the last two trajectories
        np.testing.assert_equal(dim.ra_itraj_linear[[1, 2], :30], np.concatenate((self.data[1], self.data[2]))[:30])

    def test_slice_random_access_jagged(self):
        dim = DataInMemory(self.data)

        all_data = dim.ra_itraj_jagged[:, :, :]
        for idx in range(3):
            np.testing.assert_equal(all_data[idx], self.data[idx])

        jagged = dim.ra_itraj_jagged[:, :30]
        for idx in range(3):
            np.testing.assert_equal(jagged[idx], self.data[idx][:30])

        jagged_last_dim = dim.ra_itraj_jagged[:, :, -1]
        for idx in range(3):
            np.testing.assert_equal(jagged_last_dim[idx], self.data[idx][:, -1])

    def test_iterator_context(self):
        dim = DataInMemory(np.array([1]))

        ctx = dim.iterator(stride=1).state
        assert ctx.stride == 1
        assert ctx.uniform_stride
        assert ctx.is_stride_sorted()
        assert ctx.traj_keys is None

        ctx = dim.iterator(stride=np.asarray([[0, 0], [0, 1], [0, 2]])).state
        assert not ctx.uniform_stride
        assert ctx.is_stride_sorted()
        np.testing.assert_array_equal(ctx.traj_keys, np.array([0]))

        # require sorted random access
        dim._needs_sorted_random_access_stride = True

        # sorted within trajectory, not sorted by trajectory key
        with self.assertRaises(ValueError):
            dim.iterator(stride=np.asarray([[1, 1], [1, 2], [1, 3], [0, 0], [0, 1], [0, 2]]))

        # sorted by trajectory key, not within trajectory
        with self.assertRaises(ValueError):
            dim.iterator(stride=np.asarray([[0, 0], [0, 1], [0, 2], [1, 1], [1, 5], [1, 3]]))

        np.testing.assert_array_equal(ctx.ra_indices_for_traj(0), np.array([0, 1, 2]))

    def test_data_in_memory_random_access(self):
        # access with a chunk_size that is larger than the largest index list of stride
        data_in_memory = coor.source(self.data, chunk_size=10)
        out1 = data_in_memory.get_output(stride=self.stride)

        # access with a chunk_size that is smaller than the largest index list of stride
        data_in_memory = coor.source(self.data, chunk_size=1)
        out2 = data_in_memory.get_output(stride=self.stride)

        # access in full trajectory mode
        data_in_memory = coor.source(self.data, chunk_size=0)
        out3 = data_in_memory.get_output(stride=self.stride)

        for idx in np.unique(self.stride[:, 0]):
            np.testing.assert_array_almost_equal(self.data[idx][self.stride[self.stride[:, 0] == idx][:, 1]], out1[idx])
            np.testing.assert_array_almost_equal(out1[idx], out2[idx])
            np.testing.assert_array_almost_equal(out2[idx], out3[idx])

    def test_data_in_memory_without_first_two_trajs(self):
        data_in_memory = coor.source(self.data, chunk_size=10)
        out = data_in_memory.get_output(stride=self.stride2)
        np.testing.assert_array_almost_equal(out[2], [self.data[2][0]])

    def test_csv_filereader_random_access(self):
        tmpfiles = [tempfile.mktemp(suffix='.dat') for _ in range(0, len(self.data))]
        try:
            for idx, tmp in enumerate(tmpfiles):
                np.savetxt(tmp, self.data[idx])

            # large enough chunksize
            csv_fr = coor.source(tmpfiles, chunk_size=10)
            out1 = csv_fr.get_output(stride=self.stride)

            # small chunk size
            np_fr = coor.source(tmpfiles, chunk_size=1)
            out2 = np_fr.get_output(stride=self.stride)

            for idx in np.unique(self.stride[:, 0]):
                np.testing.assert_array_almost_equal(self.data[idx][self.stride[self.stride[:, 0] == idx][:, 1]],
                                                     out1[idx])
                np.testing.assert_array_almost_equal(out1[idx], out2[idx])
        finally:
            for tmp in tmpfiles:
                try:
                    os.unlink(tmp)
                except EnvironmentError:
                    pass

    def test_numpy_filereader_random_access(self):
        tmpfiles = [tempfile.mktemp(suffix='.npy') for _ in range(0, len(self.data))]
        try:
            for idx, tmp in enumerate(tmpfiles):
                np.save(tmp, self.data[idx])
            # large enough chunk size
            np_fr = coor.source(tmpfiles, chunk_size=10)
            out1 = np_fr.get_output(stride=self.stride)

            # small chunk size
            np_fr = coor.source(tmpfiles, chunk_size=1)
            out2 = np_fr.get_output(stride=self.stride)

            # full traj mode
            np_fr = coor.source(tmpfiles, chunk_size=0)
            out3 = np_fr.get_output(stride=self.stride)

            for idx in np.unique(self.stride[:, 0]):
                np.testing.assert_array_almost_equal(self.data[idx][self.stride[self.stride[:, 0] == idx][:, 1]],
                                                     out1[idx])
                np.testing.assert_array_almost_equal(out1[idx], out2[idx])
                np.testing.assert_array_almost_equal(out2[idx], out3[idx])

        finally:
            for tmp in tmpfiles:
                try:
                    os.unlink(tmp)
                except EnvironmentError:
                    pass

    def test_transformer_iterator_random_access(self):
        kmeans = coor.cluster_kmeans(self.data, k=2)
        kmeans.in_memory = True

        for cs in range(0, 5):
            kmeans.chunksize = cs
            ref_stride = {0: 0, 1: 0, 2: 0}
            it = kmeans.iterator(stride=self.stride)
            for x in it:
                ref_stride[x[0]] += len(x[1])
            for key in list(ref_stride.keys()):
                expected = len(it.ra_indices_for_traj(key))
                assert ref_stride[key] == expected, \
                    "Expected to get exactly %s elements of trajectory %s, but got %s for chunksize=%s" \
                    % (expected, key, ref_stride[key], cs)

    def test_feature_reader_random_access_xtc(self):
        _test_ra_with_format('.xtc', self.stride)

    def test_feature_reader_random_access_dcd(self):
        _test_ra_with_format('.dcd', self.stride)

    def test_feature_reader_random_access_trr(self):
        _test_ra_with_format('.trr', self.stride)

    def test_feature_reader_random_access_hdf5(self):
        _test_ra_with_format('.h5', self.stride)

    def test_feature_reader_random_access_xyz(self):
        _test_ra_with_format('.xyz', self.stride)

    @unittest.skip("gro has no len()")
    def test_feature_reader_random_access_gro(self):
        _test_ra_with_format('.gro', self.stride)

    def test_feature_reader_random_access_netcdf(self):
        _test_ra_with_format('.nc', self.stride)

    @unittest.skip("lammpstrj has no len()")
    def test_feature_reader_random_access_lampstr(self):
        _test_ra_with_format('.lammpstrj', self.stride)

    def test_fragmented_reader_random_access(self):
        with TemporaryDirectory() as td:
            trajfiles = []
            for i in range(3):
                trajfiles.append(create_traj(start=i * 10, dir=td, length=20)[0])
            topfile = get_top()

            trajfiles = [trajfiles[0], (trajfiles[0], trajfiles[1]), trajfiles[2]]

            source = coor.source(trajfiles, top=topfile)
            assert isinstance(source, FragmentedTrajectoryReader)

            for chunksize in [0, 2, 3, 100000]:
                out = source.get_output(stride=self.stride, chunk=chunksize)
                keys = np.unique(self.stride[:, 0])
                for i, coords in enumerate(out):
                    if i in keys:
                        traj = mdtraj.load(trajfiles[i], top=topfile)
                        np.testing.assert_equal(coords,
                                                traj.xyz[
                                                    np.array(self.stride[self.stride[:, 0] == i][:, 1])
                                                ].reshape(-1, 3 * 3))

    def test_fragmented_reader_random_access1(self):
        with TemporaryDirectory() as td:
            trajfiles = []
            for i in range(3):
                trajfiles.append(create_traj(start=i * 10, dir=td, length=20)[0])
            topfile = get_top()
            trajfiles = [(trajfiles[0], trajfiles[1]), trajfiles[0],  trajfiles[2]]

            source = coor.source(trajfiles, top=topfile)
            assert isinstance(source, FragmentedTrajectoryReader)

            for r in source._readers:
                if not isinstance(r, (list, tuple)):
                    r = r[0]
                for _r in r:
                    _r._return_traj_obj = True

            from collections import defaultdict
            for chunksize in [0, 2, 3, 100000]:
                frames = defaultdict(list)
                with source.iterator(chunk=chunksize, return_trajindex=True, stride=self.stride) as it:
                    for itraj, t in it:
                        frames[itraj].append(t)

                dest = []
                for itraj in frames.keys():
                    dest.append(frames[itraj][0])

                    for t in frames[itraj][1:]:
                        dest[-1] = dest[-1].join(t)

                keys = np.unique(self.stride[:, 0])
                for i, coords in enumerate(dest):
                    if i in keys:
                        traj = mdtraj.load(trajfiles[i], top=topfile)
                        np.testing.assert_equal(coords.xyz,
                                                traj.xyz[
                                                    np.array(self.stride[self.stride[:, 0] == i][:, 1])
                                                ], err_msg="not equal for chunksize=%s" % chunksize)

if __name__ == '__main__':
    unittest.main()
