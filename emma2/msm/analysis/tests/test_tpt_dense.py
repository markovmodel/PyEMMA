r"""Unit test for the TPT-functions of the analysis API

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np

from birth_death_chain import BirthDeathChain

from emma2.msm.analysis import tpt_flux, tpt_netflux, tpt_totalflux, tpt_rate, tpt

class TestTPT(unittest.TestCase):
    def setUp(self):
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1

        self.A=[0,1]
        self.B=[8,9]
        self.a=1
        self.b=8

        self.bdc=BirthDeathChain(q, p)
        self.T=self.bdc.transition_matrix()

        """Compute mu, qminus, qplus in constructor"""
        self.tpt=tpt(self.T, self.A, self.B)

        """Use precomputed mu, qminus, qplus"""        
        self.mu=self.bdc.stationary_distribution()
        self.qminus=self.bdc.committor_backward(self.a, self.b)
        self.qplus=self.bdc.committor_forward(self.a, self.b)
        self.tpt_fast=tpt(self.T, self.A, self.B,\
                              mu=self.mu, qminus=self.qminus,\
                              qplus=self.qplus)

    def test_flux(self):
        flux=self.bdc.flux(self.a, self.b)        
        
        fluxn=self.tpt.get_flux()
        self.assertTrue(np.allclose(fluxn, flux))

        fluxn=self.tpt_fast.get_flux()
        self.assertTrue(np.allclose(fluxn, flux))

    def test_netflux(self):
        netflux=self.bdc.netflux(self.a, self.b)
        
        netfluxn=self.tpt.get_netflux()
        self.assertTrue(np.allclose(netfluxn, netflux))

        netfluxn=self.tpt_fast.get_netflux()
        self.assertTrue(np.allclose(netfluxn, netflux))        

    def test_totalflux(self):
        F=self.bdc.totalflux(self.a, self.b)

        Fn=self.tpt.get_totalflux()
        self.assertTrue(np.allclose(Fn, F))

        Fn=self.tpt_fast.get_totalflux()
        self.assertTrue(np.allclose(Fn, F))

    def test_rate(self):
        k=self.bdc.rate(self.a, self.b)
        
        kn=self.tpt.get_rate()
        self.assertTrue(np.allclose(kn, k))

        kn=self.tpt_fast.get_rate()
        self.assertTrue(np.allclose(kn, k))

    def test_backward_committor(self):
        qminus=self.qminus

        qminusn=self.tpt.get_backward_committor()
        self.assertTrue(np.allclose(qminusn, qminus))

        qminusn=self.tpt_fast.get_backward_committor()
        self.assertTrue(np.allclose(qminusn, qminus))

    def test_forward_committor(self):
        qplus=self.qplus

        qplusn=self.tpt.get_forward_committor()
        self.assertTrue(np.allclose(qplusn, qplus))

        qplusn=self.tpt_fast.get_forward_committor()
        self.assertTrue(np.allclose(qplusn, qplus))

    def test_stationary_distribution(self):
        mu=self.mu
        
        mun=self.tpt.get_stationary_distribution()
        self.assertTrue(np.allclose(mun, mu))

        mun=self.tpt_fast.get_stationary_distribution()
        self.assertTrue(np.allclose(mun, mu))


class TestTptFunctions(unittest.TestCase):
    def setUp(self):
        p=np.zeros(10)
        q=np.zeros(10)
        p[0:-1]=0.5
        q[1:]=0.5
        p[4]=0.01
        q[6]=0.1

        self.A=[0,1]
        self.B=[8,9]
        self.a=1
        self.b=8

        self.bdc=BirthDeathChain(q, p)
        self.T=self.bdc.transition_matrix()    

        self.mu=self.bdc.stationary_distribution()
        self.qminus=self.bdc.committor_backward(self.a, self.b)
        self.qplus=self.bdc.committor_forward(self.a, self.b)
    
    def test_tpt_flux(self):
        flux=self.bdc.flux(self.a, self.b)        
        
        fluxn=tpt_flux(self.T, self.A, self.B)
        self.assertTrue(np.allclose(fluxn, flux))

        fluxn=tpt_flux(self.T, self.A, self.B, mu=self.mu,\
                           qminus=self.qminus, qplus=self.qplus)
        self.assertTrue(np.allclose(fluxn, flux))

    def test_tpt_netflux(self):
        netflux=self.bdc.netflux(self.a, self.b)

        netfluxn=tpt_netflux(self.T, self.A, self.B)
        self.assertTrue(np.allclose(netfluxn, netflux))

        netfluxn=tpt_netflux(self.T, self.A, self.B, mu=self.mu,\
                           qminus=self.qminus, qplus=self.qplus)
        self.assertTrue(np.allclose(netfluxn, netflux))

    def test_tpt_totalflux(self):
        totalflux=self.bdc.totalflux(self.a, self.b)

        totalfluxn=tpt_totalflux(self.T, self.A, self.B)
        self.assertTrue(np.allclose(totalfluxn, totalflux))

        totalfluxn=tpt_totalflux(self.T, self.A, self.B, mu=self.mu,\
                           qminus=self.qminus, qplus=self.qplus)
        self.assertTrue(np.allclose(totalfluxn, totalflux))

    def test_tpt_rate(self):
        rate=self.bdc.rate(self.a, self.b)

        raten=tpt_rate(self.T, self.A, self.B)
        self.assertTrue(np.allclose(raten, rate))

        raten=tpt_rate(self.T, self.A, self.B, mu=self.mu,\
                           qminus=self.qminus, qplus=self.qplus)
        self.assertTrue(np.allclose(raten, rate))        


if __name__ == "__main__":
    unittest.main()
