r"""This module contains unit tests for the trajectory module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import os
import unittest

import numpy as np

import pyemma.util.discrete_trajectories as dt

from os.path import abspath, join
from os import pardir

testpath = abspath(join(abspath(__file__), pardir)) + '/data/'

class TestReadDiscreteTrajectory(unittest.TestCase):

    def setUp(self):
        self.filename= testpath +'dtraj.dat'

    def tearDown(self):
        pass

    def test_read_discrete_trajectory(self):
        dtraj_np=np.loadtxt(self.filename, dtype=int)
        dtraj=dt.read_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_np==dtraj))

class TestWriteDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename=testpath +'out_dtraj.dat'
        self.dtraj=np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_write_discrete_trajectory(self):
        dt.write_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n=np.loadtxt(self.filename)
        self.assertTrue(np.all(dtraj_n==self.dtraj))

class TestLoadDiscreteTrajectory(unittest.TestCase):

    def setUp(self):
        self.filename=testpath +'dtraj.npy'

    def tearDown(self):
        pass

    def test_load_discrete_trajectory(self):
        dtraj_n=np.load(self.filename)
        dtraj=dt.load_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_n==dtraj))

class TestSaveDiscreteTrajectory(unittest.TestCase):

    def setUp(self):
        self.filename=testpath +'out_dtraj.npy'
        self.dtraj=np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_save_discrete_trajectory(self):
        dt.save_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n=np.load(self.filename)
        self.assertTrue(np.all(dtraj_n==self.dtraj))

class TestDiscreteTrajectoryStatistics(unittest.TestCase):
    
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_count_1(self):
        S = np.array([0, 0, 0, 0, 0, 0])
        H = np.array([6])
        assert(dt.number_of_states(S) == 1)
        assert(dt.number_of_states(S, only_used=True) == 1)
        assert(np.allclose(dt.count_states(S),H))

    def test_count_2(self):
        S = np.array([1, 1, 1, 1, 1, 1])
        H = np.array([0,6])
        assert(dt.number_of_states(S) == 2)
        assert(dt.number_of_states(S, only_used=True) == 1)
        assert(np.allclose(dt.count_states(S),H))

    def test_count_3(self):
        S1 = np.array([0, 1, 2, 3, 4])
        S2 = np.array([2, 2, 2, 2, 6])
        H = np.array([1, 1, 5, 1, 1, 0, 1])
        assert(dt.number_of_states([S1,S2]) == 7)
        assert(dt.number_of_states([S1,S2], only_used=True) == 6)
        assert(np.allclose(dt.count_states([S1,S2]),H))

    def test_count_big(self):
        dtraj = dt.read_discrete_trajectory(testpath+'2well_traj_100K.dat')
        # just run these to see if there's any exception
        dt.number_of_states(dtraj)
        dt.count_states(dtraj)

class TestIndexStates(unittest.TestCase):

    def test_subset_error(self):
        dtraj =[0,1,2,3,2,1,0]
        # should be a ValueError because this is not a subset
        with self.assertRaises(ValueError):
            dt.index_states(dtraj, subset=[3,4,5])

    def test_onetraj(self):
        dtraj =[0,1,2,3,2,1,0]
        # should be a ValueError because this is not a subset
        res = dt.index_states(dtraj)
        expected = [np.array([[0,0],[0,6]]),np.array([[0,1],[0,5]]),np.array([[0,2],[0,4]]),np.array([[0,3]])]
        assert(len(res) == len(expected))
        for i in range(len(res)):
            assert(res[i].shape == expected[i].shape)
            assert(np.alltrue(res[i] == expected[i]))

    def test_onetraj_sub(self):
        dtraj =[0,1,2,3,2,1,0]
        # should be a ValueError because this is not a subset
        res = dt.index_states(dtraj, subset=[2,3])
        expected = [np.array([[0,2],[0,4]]),np.array([[0,3]])]
        assert(len(res) == len(expected))
        for i in range(len(res)):
            assert(res[i].shape == expected[i].shape)
            assert(np.alltrue(res[i] == expected[i]))

    def test_twotraj(self):
        dtrajs = [[0,1,2,3,2,1,0], [3,4,5]]
        # should be a ValueError because this is not a subset
        res = dt.index_states(dtrajs)
        expected = [np.array([[0,0],[0,6]]),np.array([[0,1],[0,5]]),np.array([[0,2],[0,4]]),np.array([[0,3],[1,0]]),np.array([[1,1]]),np.array([[1,2]])]
        assert(len(res) == len(expected))
        for i in range(len(res)):
            assert(res[i].shape == expected[i].shape)
            assert(np.alltrue(res[i] == expected[i]))

    def test_big(self):
        dtraj = dt.read_discrete_trajectory(testpath+'2well_traj_100K.dat')
        # just run these to see if there's any exception
        dt.index_states(dtraj)

class TestSampleIndexes(unittest.TestCase):

    def test_sample_by_sequence(self):
        dtraj =[0,1,2,3,2,1,0]
        idx = dt.index_states(dtraj)
        seq = [0,1,1,1,0,0,0,0,1,1]
        sidx = dt.sample_indexes_by_sequence(idx, seq)
        assert(np.alltrue(sidx.shape == (len(seq),2)))
        for t in range(sidx.shape[0]):
            assert(sidx[t,0] == 0) # did we pick the right traj?
            assert(dtraj[sidx[t,1]] == seq[t]) # did we pick the right states?

    def test_sample_by_state_replace(self):
        dtraj =[0,1,2,3,2,1,0]
        idx = dt.index_states(dtraj)
        sidx = dt.sample_indexes_by_state(idx, 5)
        for i in range(4):
            assert(sidx[i].shape[0] == 5)
            for t in range(sidx[i].shape[0]):
                assert(dtraj[sidx[i][t,1]] == i)

    def test_sample_by_state_replace_subset(self):
        dtraj =[0,1,2,3,2,1,0]
        idx = dt.index_states(dtraj)
        subset = [1,2]
        sidx = dt.sample_indexes_by_state(idx, 5, subset=subset)
        for i in range(len(subset)):
            assert(sidx[i].shape[0] == 5)
            for t in range(sidx[i].shape[0]):
                assert(dtraj[sidx[i][t,1]] == subset[i])

if __name__=="__main__":
    unittest.main()
