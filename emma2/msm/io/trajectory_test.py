"""This module contains unit tests for the trajectory module"""

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
        

if __name__=="__main__":
    unittest.main()
        
