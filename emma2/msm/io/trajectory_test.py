"""This module contains unit tests for the trajectory module"""
import os
import unittest

import numpy as np

import trajectory

testpath = __package__.replace('.', '/')

class TestReadDiscreteTrajectory(unittest.TestCase):
    
    def setUp(self):
        self.filename= testpath +'/test/dtraj.dat'

    def tearDown(self):
        pass

    def test_read_discrete_trajectory(self):
        dtraj_np=np.loadtxt(self.filename, dtype=int)
        dtraj=trajectory.read_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_np==dtraj))

class TestWriteDiscreteTrajectory(unittest.TestCase):
    def setUp(self):
        self.filename=testpath +'/test/out_dtraj.dat'
        self.dtraj=np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_write_discrete_trajectory(self):
        trajectory.write_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n=np.loadtxt(self.filename)
        self.assertTrue(np.all(dtraj_n==self.dtraj))

class TestLoadDiscreteTrajectory(unittest.TestCase):
    
    def setUp(self):
        self.filename=testpath +'/test/dtraj.npy'

    def tearDown(self):
        pass

    def test_load_discrete_trajectory(self):
        dtraj_n=np.load(self.filename)
        dtraj=trajectory.load_discrete_trajectory(self.filename)
        self.assertTrue(np.all(dtraj_n==dtraj))

class TestSaveDiscreteTrajectory(unittest.TestCase):
    
    def setUp(self):
        self.filename=testpath +'/test/out_dtraj.npy'
        self.dtraj=np.arange(10000)

    def tearDown(self):
        os.remove(self.filename)

    def test_save_discrete_trajectory(self):
        trajectory.save_discrete_trajectory(self.filename, self.dtraj)
        dtraj_n=np.load(self.filename)
        self.assertTrue(np.all(dtraj_n==self.dtraj))

if __name__=="__main__":
    unittest.main()
