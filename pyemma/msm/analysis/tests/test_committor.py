r"""Unit tests for the committor API-function

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np

from pyemma.msm.analysis import committor

from birth_death_chain import BirthDeathChain

class TestCommittorDense(unittest.TestCase):
    def setUp(self):
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1

        self.bdc=BirthDeathChain(q, p)

    def tearDown(self):
        pass

    def test_forward_comittor(self):
        P=self.bdc.transition_matrix()
        un=committor(P, [0, 1], [8, 9], forward=True)
        u=self.bdc.committor_forward(1, 8)              
        self.assertTrue(np.allclose(un, u))

    def test_backward_comittor(self):
        P=self.bdc.transition_matrix()
        un=committor(P, [0, 1], [8, 9], forward=False)
        u=self.bdc.committor_backward(1, 8)        
        self.assertTrue(np.allclose(un, u))
        
class TestCommittorSparse(unittest.TestCase):
    def setUp(self):
        p=np.zeros(100)
        q=np.zeros(100)
        p[0:-1]=0.5
        q[1:]=0.5
        p[49]=0.01
        q[51]=0.1

        self.bdc=BirthDeathChain(q, p)

    def tearDown(self):
        pass

    def test_forward_comittor(self):
        P=self.bdc.transition_matrix_sparse()
        un=committor(P, range(10), range(90,100), forward=True)
        u=self.bdc.committor_forward(9, 90)               
        self.assertTrue(np.allclose(un, u))

    def test_backward_comittor(self):
        P=self.bdc.transition_matrix_sparse()
        un=committor(P, range(10), range(90,100), forward=False)
        u=self.bdc.committor_backward(9, 90)               
        self.assertTrue(np.allclose(un, u))

if __name__ == "__main__":
    unittest.main()
