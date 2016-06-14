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
Created on 30.04.2015

@author: marscher
'''

from __future__ import absolute_import

from tempfile import NamedTemporaryFile

try:
    import bsddb
    have_bsddb = True
except ImportError:
    have_bsddb = False

import os
import six
import tempfile
import unittest

from pyemma.coordinates import api
from pyemma.coordinates.data.feature_reader import FeatureReader
from pyemma.coordinates.data.numpy_filereader import NumPyFileReader
from pyemma.coordinates.data.py_csv_reader import PyCSVReader
from pyemma.coordinates.data.util.traj_info_cache import TrajectoryInfoCache
from pyemma.coordinates.tests.util import create_traj
from pyemma.datasets import get_bpti_test_data
from pyemma.util import config
from pyemma.util.files import TemporaryDirectory
import mdtraj
import pkg_resources
import pyemma
import numpy as np

if six.PY2:
    import dumbdbm
    import mock
else:
    from dbm import dumb as dumbdbm
    from unittest import mock

xtcfiles = get_bpti_test_data()['trajs']
pdbfile = get_bpti_test_data()['top']


class TestTrajectoryInfoCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.work_dir = tempfile.mkdtemp("traj_cache_test")

    def setUp(self):
        self.tmpfile = tempfile.mktemp(dir=self.work_dir)
        self.db = TrajectoryInfoCache(self.tmpfile)

        assert len(self.db._database) == 1, len(self.db._database)
        assert 'db_version' in self.db._database
        assert int(self.db._database['db_version']) >= 1

    def tearDown(self):
        del self.db

    @classmethod
    def tearDownClass(cls):
        import shutil
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_get_instance(self):
        # test for exceptions in singleton creation
        inst = TrajectoryInfoCache.instance()
        inst.current_db_version

    @unittest.skip("persistence currently disabled.")
    def test_store_load_traj_info(self):
        x = np.random.random((10, 3))
        my_conf = config()
        my_conf.cfg_dir = self.work_dir
        with mock.patch('pyemma.coordinates.data.util.traj_info_cache.config', my_conf):
            with NamedTemporaryFile(delete=False) as fh:
                np.savetxt(fh.name, x)
                reader = api.source(fh.name)
                info = self.db[fh.name, reader]
                self.db._database.close()
                self.db._database = dumbdbm.open(self.db.database_filename, 'r')
                info2 = self.db[fh.name, reader]
                self.assertEqual(info2, info)

    def test_exceptions(self):
        # in accessible files
        not_existant = ''.join(
            chr(i) for i in np.random.random_integers(65, 90, size=10)) + '.npy'
        bad = [not_existant]  # should be unaccessible or non existant
        with self.assertRaises(ValueError) as cm:
            api.source(bad)
            assert bad[0] in cm.exception.message

        # empty files
        with NamedTemporaryFile(delete=False) as f:
            f.close()
            with self.assertRaises(ValueError) as cm:
                api.source(f.name)
                assert f.name in cm.exception.message

    def test_featurereader_xtc(self):
        # cause cache failures
        config['use_trajectory_lengths_cache'] = False
        reader = FeatureReader(xtcfiles, pdbfile)
        config['use_trajectory_lengths_cache'] = True

        results = {}
        for f in xtcfiles:
            traj_info = self.db[f, reader]
            results[f] = traj_info.ndim, traj_info.length, traj_info.offsets

        expected = {}
        for f in xtcfiles:
            with mdtraj.open(f) as fh:
                length = len(fh)
                ndim = fh.read(1)[0].shape[1]
                offsets = fh.offsets if hasattr(fh, 'offsets') else []
                expected[f] = ndim, length, offsets

        np.testing.assert_equal(results, expected)

    def test_npy_reader(self):
        lengths_and_dims = [(7, 3), (23, 3), (27, 3)]
        data = [
            np.empty((n, dim)) for n, dim in lengths_and_dims]
        files = []
        with TemporaryDirectory() as td:
            for i, x in enumerate(data):
                fn = os.path.join(td, "%i.npy" % i)
                np.save(fn, x)
                files.append(fn)

            reader = NumPyFileReader(files)

            # cache it and compare
            results = {f: (self.db[f, reader].length, self.db[f, reader].ndim,
                           self.db[f, reader].offsets) for f in files}
            expected = {f: (len(data[i]), data[i].shape[1], [])
                        for i, f in enumerate(files)}
            np.testing.assert_equal(results, expected)

    def test_csvreader(self):
        data = np.random.random((101, 3))
        fn = tempfile.mktemp()
        try:
            np.savetxt(fn, data)
            # calc offsets
            offsets = [0]
            with open(fn, PyCSVReader.DEFAULT_OPEN_MODE) as new_fh:
                while new_fh.readline():
                    offsets.append(new_fh.tell())
            reader = PyCSVReader(fn)
            assert reader.dimension() == 3
            trajinfo = reader._get_traj_info(fn)
            np.testing.assert_equal(offsets, trajinfo.offsets)
        finally:
            os.unlink(fn)

    def test_fragmented_reader(self):
        top_file = pkg_resources.resource_filename(__name__, 'data/test.pdb')
        trajfiles = []
        nframes = []
        with TemporaryDirectory() as wd:
            for _ in range(3):
                f, _, l = create_traj(top_file, dir=wd)
                trajfiles.append(f)
                nframes.append(l)
            # three trajectories: one consisting of all three, one consisting of the first,
            # one consisting of the first and the last
            reader = api.source(
                [trajfiles, [trajfiles[0]], [trajfiles[0], trajfiles[2]]], top=top_file)
            np.testing.assert_equal(reader.trajectory_lengths(),
                                    [sum(nframes), nframes[0], nframes[0] + nframes[2]])

    def test_feature_reader_xyz(self):
        traj = mdtraj.load(xtcfiles, top=pdbfile)
        length = len(traj)

        with NamedTemporaryFile(mode='wb', suffix='.xyz', delete=False) as f:
            fn = f.name
            traj.save_xyz(fn)
            f.close()
            reader = pyemma.coordinates.source(fn, top=pdbfile)
            self.assertEqual(reader.trajectory_length(0), length)

    def test_data_in_mem(self):
        # make sure cache is not used for data in memory!
        data = [np.empty((3, 3))] * 3
        api.source(data)
        assert len(self.db._database) == 1

    def test_old_db_conversion(self):
        # prior 2.1, database only contained lengths (int as string) entries
        # check conversion is happening
        with NamedTemporaryFile(suffix='.npy', delete=False) as f:
            db = TrajectoryInfoCache(None)
            fn = f.name
            np.save(fn, [1, 2, 3])
            f.close() # windows sucks
            reader = api.source(fn)
            hash = db._get_file_hash(fn)
            db._database = {hash: str(3)}

            info = db[fn, reader]
            assert info.length == 3
            assert info.ndim == 1
            assert info.offsets == []

    def test_corrupted_db(self):
        with NamedTemporaryFile(mode='w', suffix='.dat', delete=False) as f:
            f.write("makes no sense!!!!")
            f.close()
        name = f.name
        db = TrajectoryInfoCache(name)

        # ensure we can perform lookups on the broken db without exception.
        r = api.source(xtcfiles[0], top=pdbfile)
        db[xtcfiles[0], r]

if __name__ == "__main__":
    unittest.main()
