r"""Unit test for the fingerprint module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest

import numpy as np

from decomposition import rdl_decomposition, timescales

from committor_test import BirthDeathChain

from fingerprints import fingerprint_correlation, fingerprint_relaxation

class TestFingerprintCorrelation(unittest.TestCase):

    def setUp(self):
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1

        self.k=3
        self.bdc=BirthDeathChain(q, p)
        
        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix_sparse()
        R, D, L=rdl_decomposition(self.T, k=self.k)
        self.L=L
        self.ts=timescales(self.T, k=self.k)

        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1=obs1
        self.obs2=obs2
        self.one_vec=np.ones(10)

        """Eectation amplitudes"""
        expectation=np.dot(self.mu, self.obs1)
        self.exp_amp=np.zeros(self.k)
        self.exp_amp[0]=expectation

        """Autocorrelation amplitudes"""
        tmp=np.dot(self.L, self.obs1) 
        self.acorr_amp=tmp*tmp

        """Crosscorrelation amplitudes"""
        tmp1=np.dot(self.L, self.obs1)
        tmp2=np.dot(self.L, self.obs2)
        self.corr_amp=tmp1*tmp2

    def test_expectation(self):
        tsn, exp_ampn=fingerprint_correlation(self.T, self.obs1, obs2=self.one_vec, k=self.k)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(exp_ampn, self.exp_amp))        

    def test_autocorrelation(self):
        tsn, acorr_ampn=fingerprint_correlation(self.T, self.obs1, k=self.k)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(acorr_ampn, self.acorr_amp))   

    def test_crosscorrelation(self):
        tsn, corr_ampn=fingerprint_correlation(self.T, self.obs1, obs2=self.obs2, k=self.k)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(corr_ampn, self.corr_amp))   

class TestFingerprintRelaxation(unittest.TestCase):

    def setUp(self):
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1

        self.k=3
        self.bdc=BirthDeathChain(q, p)
        
        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix_sparse()
        R, D, L=rdl_decomposition(self.T, k=self.k)
        self.L=L
        self.R=R
        self.ts=timescales(self.T, k=self.k)


        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1=obs1
        self.obs2=obs2
        self.one_vec=np.ones(10)

        w0=np.zeros(10)
        w0[0:4]=0.25
        self.p0=w0
        

        """Expectation"""
        tmp1=np.dot(self.p0, self.R)
        tmp2=np.dot(self.L, self.obs1)
        self.exp_amp=tmp1*tmp2

    def test_expectation(self):
        tsn, exp_ampn=fingerprint_relaxation(self.T, self.p0, self.obs1, k=self.k)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(exp_ampn, self.exp_amp))        

if __name__ == "__main__":
    unittest.main()
