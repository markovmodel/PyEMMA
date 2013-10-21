"""This module contains unit tests for the trajectory module"""
import os
import unittest

import numpy as np
import scipy.sparse 

import trajectory

class TestReadDiscreteTrajectorySingle(unittest.TestCase):
    
    def setUp(self):
        self.name='test/dtraj1.dat'
        # self.name2='test/dtraj2.dat'

    def tearDown(self):
        pass

    def test_read_discrete_trajectory_single(self):
        dtraj_np=np.loadtxt(self.name, dtype=int)
        dtraj=trajectory.read_discrete_trajectory_single(self.name)
        self.assertTrue(np.all(dtraj_np==dtraj))

class TestReadDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.name1='test/dtraj1.dat'
        self.name2='test/dtraj2.dat'
        self.names=[self.name1, self.name2]
        self.dtraj1_np=np.loadtxt(self.name1, dtype=int)
        self.dtraj2_np=np.loadtxt(self.name2, dtype=int)
        self.dtraj_np=[self.dtraj1_np, self.dtraj2_np]


    def tearDown(self):
        pass

    def test_read_discrete_trajectory(self):
        dtraj=trajectory.read_discrete_trajectory(self.name1)
        self.assertTrue(np.all(self.dtraj1_np==dtraj))

        dtraj=trajectory.read_discrete_trajectory(self.names)
        for i in range(len(dtraj)):
            self.assertTrue(np.all(self.dtraj_np[i]==dtraj[i]))

class TestWriteDiscreteTrajectorySingle(unittest.TestCase):
    def setUp(self):
        self.filename='test/out_dtraj.dat'
        self.dtraj=np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_write_discrete_trajectory_single(self):
        trajectory.write_discrete_trajectory_single(self.filename, self.dtraj)
        dtraj_n=np.loadtxt(self.filename)
        self.assertTrue(np.all(dtraj_n==self.dtraj))

class TestWriteDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename1='test/out_dtraj1.dat'
        self.dtraj1=np.arange(10000)
        self.filename2='test/out_dtraj2.dat'
        self.dtraj2=np.arange(20000)    
        
        self.filenames=[self.filename1, self.filename2]
        self.dtraj=[self.dtraj1, self.dtraj2]

    def tearDown(self):
        os.remove(self.filename1)
        os.remove(self.filename2)

    def test_write_discrete_trajectory(self):
        trajectory.write_discrete_trajectory(self.filename1, self.dtraj1)
        dtraj_n=np.loadtxt(self.filename1)
        self.assertTrue(np.all(dtraj_n==self.dtraj1))

        trajectory.write_discrete_trajectory(self.filenames, self.dtraj)
        for i in range(len(self.filenames)):
            dtraj_n=np.loadtxt(self.filenames[i])
            self.assertTrue(np.all(dtraj_n==self.dtraj[i]))

if __name__=="__main__":
    unittest.main()
        
