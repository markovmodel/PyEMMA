'''
Created on 19.01.2015

@author: marscher
'''
import unittest
from coordinates.coordinate_transformation.discretizer import Discretizer


class TestDiscretizer(unittest.TestCase):

    def testName(self):
        d = Discretizer(['/home/marscher/kchan/traj01_sliced.xtc'],
                        # '/home/marscher/kchan/Traj01.xtc'],
                       #  '/home/marscher/kchan/Traj_link.xtc'],
                        '/home/marscher/kchan/Traj_Structure.pdb')
        d.run()


if __name__ == "__main__":
    unittest.main()
