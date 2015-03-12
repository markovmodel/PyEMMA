'''
Created on 06.12.2013

@author: jan-hendrikprinz

This module provides the unittest for the pcca module

'''
import unittest
import numpy as np
from scipy import sparse

from pyemma.util.numeric import assert_allclose
from pcca import pcca


class TestPCCA(unittest.TestCase):


    def setUp(self):
        pass

    def test_pcca_no_transition_matrix(self):
        P = np.array([[1.0, 1.0],
                      [0.1, 0.9]])
        try:
            pcca(P, 2)
            # no ValueError? then fail.
            assert False
        except ValueError:
            pass
        except: # different exception
            assert False

    def test_pcca_no_detailed_balance(self):
        P = np.array([[0.8, 0.1, 0.1],
                      [0.3, 0.2, 0.5],
                      [0.6, 0.3, 0.1]])
        try:
            pcca(P, 2)
            # no ValueError? then fail.
            assert False
        except ValueError:
            pass
        except: # different exception
            assert False


    def test_pcca_1(self):
        P = np.array([[1, 0],
                      [0, 1]])
        chi = pcca(P, 2)
        sol = np.array([[ 1.,  0.],
                        [ 0.,  1.]])
        assert_allclose(chi, sol)


    def test_pcca_2(self):
        P = np.array([[0.0, 1.0, 0.0],
                      [0.0, 0.999, 0.001],
                      [0.0, 0.001, 0.999]])
        chi = pcca(P, 2)
        sol = np.array([[ 1.,  0.],
                        [ 1.,  0.],
                        [ 0.,  1.]])
        assert_allclose(chi, sol)


    def test_pcca_3(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        # n=2
        chi = pcca(P, 2)
        sol = np.array([[ 1.,  0.],
                        [ 1.,  0.],
                        [ 0.,  1.],
                        [ 0.,  1.]])
        assert_allclose(chi, sol)
        # n=3
        chi = pcca(P, 3)
        sol = np.array([[ 1.,  0.,  0.],
                        [ 0.,  1.,  0.],
                        [ 0.,  0.,  1.],
                        [ 0.,  0.,  1.]])
        assert_allclose(chi, sol)
        # n=4
        chi = pcca(P, 4)
        sol = np.array([[ 1.,  0.,  0.,  0.],
                        [ 0.,  1.,  0.,  0.],
                        [ 0.,  0.,  1.,  0.],
                        [ 0.,  0.,  0.,  1.]])
        assert_allclose(chi, sol)


    def test_pcca_4(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0],
                      [0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])        
        chi = pcca(P, 2)
        sol = np.array([[ 1.,  0.],
                        [ 1.,  0.],
                        [ 1.,  0.],
                        [ 0.,  1.]])
        assert_allclose(chi, sol)


    def test_pcca_5(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.0, 0.2, 0.8]])
        # n=2
        chi = pcca(P, 2)
        sol = np.array([[ 1. ,  0. ],
                        [ 1. ,  0. ],
                        [ 0.5,  0.5],
                        [ 0. ,  1. ],
                        [ 0. ,  1. ]])
        assert_allclose(chi, sol)
        # n=4
        chi = pcca(P, 4)
        sol = np.array([[ 1. ,  0. ,  0. ,  0. ],
                        [ 0. ,  1. ,  0. ,  0. ],
                        [ 0. ,  0.5,  0.5,  0. ],
                        [ 0. ,  0. ,  1. ,  0. ],
                        [ 0. ,  0. ,  0. ,  1. ]])
        assert_allclose(chi, sol)


    def test_pcca_large(self):
        import os
        P = np.loadtxt(os.path.split(__file__)[0]+'/../tests/P_rev_251x251.dat')
        # n=2
        chi = pcca(P, 2)
        assert(np.alltrue(chi >= 0))
        assert(np.alltrue(chi <= 1))
        # n=3
        chi = pcca(P, 3)
        assert(np.alltrue(chi >= 0))
        assert(np.alltrue(chi <= 1))
        # n=4
        chi = pcca(P, 4)
        assert(np.alltrue(chi >= 0))
        assert(np.alltrue(chi <= 1))


if __name__ == "__main__":
    unittest.main()
