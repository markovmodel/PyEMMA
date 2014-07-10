r"""Unit test for the fingerprint module

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest

import numpy as np

from decomposition import rdl_decomposition, timescales

from committor_test import BirthDeathChain

from fingerprints import fingerprint, evaluate_fingerprint, correlation, relaxation, expectation

class TestFingerprintEquilibrium(unittest.TestCase):
    def setUp(self):
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1

        self.bdc=BirthDeathChain(q, p)
        
        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix()
        R, D, L=rdl_decomposition(self.T)
        self.L=L
        self.ts=timescales(self.T)
        self.times=np.array([1, 5, 10, 20])

        ev=np.diagonal(D)
        self.ev_t=np.e**(-self.times[:,np.newaxis]/self.ts[np.newaxis,:])


        obs1 = np.zeros(10)
        obs1[0] = 1
        obs1[1] = 1
        obs2 = np.zeros(10)
        obs2[8] = 1
        obs2[9] = 1

        self.obs1=obs1
        self.obs2=obs2
        self.one_vec=np.ones(10)

        """Expectation"""
        self.exp=np.dot(self.mu, self.obs1)

        """Autocorrelation amplitudes"""
        tmp=np.dot(self.L, self.obs1) 
        self.acorr_amp=tmp*tmp
        self.acorr=np.dot(self.ev_t, self.acorr_amp)

        """Crosscorrelation amplitudes"""
        tmp1=np.dot(self.L, self.obs1)
        tmp2=np.dot(self.L, self.obs2)
        self.corr_amp=tmp1*tmp2
        self.corr=np.dot(self.ev_t, self.corr_amp)      

    def test_autocorrelation_fingerprint(self):
        tsn, acorr_ampn=fingerprint(self.T, self.obs1)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(acorr_ampn, self.acorr_amp))

    def test_crosscorrelation_fingerprint(self):
        tsn, corr_ampn=fingerprint(self.T, self.obs1, obs2=self.obs2)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(corr_ampn, self.corr_amp))  

    def test_correlation(self):
        """Auto-correlation"""
        acorrn=correlation(self.T, self.obs1, times=self.times)
        self.assertTrue(np.allclose(acorrn, self.acorr))
    
        """Cross-correlation"""
        corrn=correlation(self.T, self.obs1, obs2=self.obs2, times=self.times)
        self.assertTrue(np.allclose(corrn, self.corr)) 

    def test_expectation(self):
        expn=expectation(self.T, self.obs1)
        self.assertTrue(np.allclose(expn, self.exp))       

class TestFingerprintPerturbation(unittest.TestCase):

    def setUp(self):
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1

        self.bdc=BirthDeathChain(q, p)
        
        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix()
        R, D, L=rdl_decomposition(self.T)
        self.L=L
        self.R=R
        self.ts=timescales(self.T)
        self.times=np.array([1, 5, 10, 20])

        ev=np.diagonal(D)
        self.ev_t=np.e**(-self.times[:,np.newaxis]/self.ts[np.newaxis,:])


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

        """Relaxation"""
        tmp1=np.dot(self.p0, self.R)
        tmp2=np.dot(self.L, self.obs1)
        self.exp_amp=tmp1*tmp2
        self.exp=np.dot(self.ev_t, self.exp_amp)
        
        """Autocorrelation"""
        tmp1=np.dot(self.p0*self.obs1, self.R)
        tmp2=np.dot(self.L, self.obs1)
        self.acorr_amp=tmp1*tmp2
        self.acorr=np.dot(self.ev_t, self.acorr_amp)

        """Cross-correlation"""
        tmp1=np.dot(self.p0*self.obs1, self.R)
        tmp2=np.dot(self.L, self.obs2)
        self.corr_amp=tmp1*tmp2
        self.corr=np.dot(self.ev_t, self.corr_amp)

    def test_relaxation_fingerprint(self):
        tsn, exp_ampn=fingerprint(self.T, self.one_vec, obs2=self.obs1, p0=self.p0)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(exp_ampn, self.exp_amp))    

    def test_autocorrelation_fingerprint(self):
        tsn, acorr_ampn=fingerprint(self.T, self.obs1, p0=self.p0)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(acorr_ampn, self.acorr_amp)) 

    def test_crosscorrelation_fingerprint(self):
        tsn, corr_ampn=fingerprint(self.T, self.obs1, obs2=self.obs2, p0=self.p0)
        self.assertTrue(np.allclose(tsn, self.ts))
        self.assertTrue(np.allclose(corr_ampn, self.corr_amp)) 

    def test_relaxation(self):
        expn=relaxation(self.T, self.p0, self.obs1, times=self.times)
        self.assertTrue(np.allclose(expn, self.exp))             
  
class TestFingerprintEvaluation(unittest.TestCase):
    def setUp(self):
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1

        self.bdc=BirthDeathChain(q, p)
        
        self.mu = self.bdc.stationary_distribution()
        self.T = self.bdc.transition_matrix()
        R, D, L=rdl_decomposition(self.T)
        self.L=L
        self.R=R
        self.ts=timescales(self.T)            
        self.times=np.array([1, 5, 10, 20])

        ev=np.diagonal(D)
        self.ev_t=np.e**(-self.times[:,np.newaxis]/self.ts[np.newaxis,:])


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

        """Expectation non-equilibrium (relaxation)"""
        tmp1=np.dot(self.p0, self.R)
        tmp2=np.dot(self.L, self.obs1)
        exp_amp=tmp1*tmp2
        self.exp=np.dot(self.ev_t, exp_amp)

    def test_evaluate(self):
        timescales, amplitudes=fingerprint(self.T, self.one_vec, obs2=self.obs1, p0=self.p0)
        expn=evaluate_fingerprint(timescales, amplitudes, self.times)
        self.assertTrue(np.allclose(expn, self.exp))




        
if __name__ == "__main__":
    unittest.main()
