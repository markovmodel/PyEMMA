'''
Created on Jun 6, 2014

@author: marscher
'''
import unittest
from emma2.msm.estimation.dense.tmatrix_sampler_jwrapper import ITransitionMatrixSampler
import numpy as np
from emma2.msm.analysis.api import stationary_distribution
from emma2.util.pystallone import ndarray_to_stallone_array

def assertSampler2x2(sampler, C, nsample, errtol):
    """
     same function as in stallone.test.mc.MarkovModelFactoryTest
    """
    c1 = float(np.sum(C[0]))
    c2 = float(np.sum(C[1]))
    n = C.shape[0]

    samplesT12 = np.empty(nsample)
    samplesT21 = np.empty(nsample)
    for i in xrange(nsample):
        T = sampler.sample(100)
        samplesT12[i] = T[0, 1]
        samplesT21[i] = T[1, 0]
    
    # check means
    true_meanT12 = float(C[0, 1] + 1) / float(c1 + n)
    sample_meanT12 = np.mean(samplesT12)
    err_T12 = np.abs(true_meanT12 - sample_meanT12) / true_meanT12
    assert float(err_T12) < errtol
    
    true_meanT21 = float(C[1, 0] + 1) / float(c2 + n)
    sample_meanT21 = np.mean(samplesT21)
    err_T21 = np.abs(true_meanT21 - sample_meanT21) / true_meanT21
    assert float(err_T21) < errtol
    
    # check variance
    true_varT12 = true_meanT12 * (1.0 - true_meanT12) / float(c1 + n + 1)
    sample_varT12 = np.var(samplesT12)
    err_varT12 = np.abs(true_varT12 - sample_varT12) / true_varT12
    assert float(err_varT12) < errtol
    
    true_varT21 = true_meanT21 * (1.0 - true_meanT21) / float(c2 + n + 1)
    sample_varT21 = np.var(samplesT21)
    err_varT21 = np.abs(true_varT21 - sample_varT21) / true_varT21
    assert float(err_varT21) < errtol

class TestJavaTransitionMatrixSampler(unittest.TestCase):
    """
    note this is very slow, since it calls nsample times jni methods.
    """

    def setUp(self):
        self.C = np.array([[5, 2 ],
                           [1, 10]])
        self.errtol = 1e-2
        self.nsample = 100000
    
    def testSamplerRev(self):
        sampler_rev = ITransitionMatrixSampler(self.C, reversible=True)
        assertSampler2x2(sampler_rev, self.C, self.nsample, self.errtol)
    
    def testSamplerNonRev(self):
        sampler_nonrev = ITransitionMatrixSampler(self.C, reversible=False)
        assertSampler2x2(sampler_nonrev, self.C, self.nsample, self.errtol)
    
    def testSamplerRevPiFix(self):
        mu = np.array([0.62921595, 0.37078405])
        sampler_rev_pi = ITransitionMatrixSampler(self.C, reversible=True, mu=mu)
        assertSampler2x2(sampler_rev_pi, self.C, self.nsample, self.errtol)

if __name__ == "__main__":
    unittest.main()
