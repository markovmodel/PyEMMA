r"""This module contains unit tests for the trajectory module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import os
import unittest

import numpy as np

from os.path import abspath, join
from os import pardir

from pyemma.msm.io import read_discrete_trajectory, write_discrete_trajectory, \
    load_discrete_trajectory, save_discrete_trajectory

testpath = abspath(join(abspath(__file__), pardir)) + '/testfiles/'


class TestReadDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'dtraj.dat'

    def tearDown(self):
        pass

    def test_read_discrete_trajectory(self):
        dtraj_np = np.loadtxt(self.filename, dtype=int)
        dtraj = read_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_np == dtraj))


class TestWriteDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'out_dtraj.dat'
        self.dtraj = np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_write_discrete_trajectory(self):
        write_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n = np.loadtxt(self.filename)
        self.assertTrue(np.all(dtraj_n == self.dtraj))


class TestLoadDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'dtraj.npy'

    def tearDown(self):
        pass

    def test_load_discrete_trajectory(self):
        dtraj_n = np.load(self.filename)
        dtraj = load_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_n == dtraj))


class TestSaveDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename = testpath + 'out_dtraj.npy'
        self.dtraj = np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_save_discrete_trajectory(self):
        save_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n = np.load(self.filename)
        self.assertTrue(np.all(dtraj_n == self.dtraj))


if __name__ == "__main__":
    unittest.main()
