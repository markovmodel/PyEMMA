r"""Unit test for the TPT-functions of the analysis API

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np

from scipy.sparse import csr_matrix

import pyemma.msm.flux as flux

################################################################################
# Dense
################################################################################

class TestTPTDense(unittest.TestCase):
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

        import pyemma.msm.analysis.dense.birth_death_chain
        self.bdc=pyemma.msm.analysis.dense.birth_death_chain.BirthDeathChain(q, p)
        self.T=self.bdc.transition_matrix()

        """Compute mu, qminus, qplus in constructor"""
        self.tpt = flux.tpt(self.T, self.A, self.B)

        """Use precomputed mu, qminus, qplus"""        
        self.mu=self.bdc.stationary_distribution()
        self.qminus=self.bdc.committor_backward(self.a, self.b)
        self.qplus=self.bdc.committor_forward(self.a, self.b)
        self.tpt_fast=flux.tpt(self.T, self.A, self.B,\
                              mu=self.mu, qminus=self.qminus,\
                              qplus=self.qplus)

    def test_grossflux(self):
        flux=self.bdc.flux(self.a, self.b)        
        
        fluxn=self.tpt.gross_flux
        self.assertTrue(np.allclose(fluxn, flux))

        fluxn=self.tpt_fast.gross_flux
        self.assertTrue(np.allclose(fluxn, flux))

    def test_netflux(self):
        netflux=self.bdc.netflux(self.a, self.b)
        
        netfluxn=self.tpt.net_flux
        self.assertTrue(np.allclose(netfluxn, netflux))

        netfluxn=self.tpt_fast.net_flux
        self.assertTrue(np.allclose(netfluxn, netflux))        

    def test_totalflux(self):
        F=self.bdc.totalflux(self.a, self.b)

        Fn=self.tpt.total_flux
        self.assertTrue(np.allclose(Fn, F))

        Fn=self.tpt_fast.total_flux
        self.assertTrue(np.allclose(Fn, F))

    def test_rate(self):
        k=self.bdc.rate(self.a, self.b)
        
        kn=self.tpt.rate
        self.assertTrue(np.allclose(kn, k))

        kn=self.tpt_fast.rate
        self.assertTrue(np.allclose(kn, k))

    def test_backward_committor(self):
        qminus=self.qminus

        qminusn=self.tpt.backward_committor
        self.assertTrue(np.allclose(qminusn, qminus))

        qminusn=self.tpt_fast.backward_committor
        self.assertTrue(np.allclose(qminusn, qminus))

    def test_forward_committor(self):
        qplus=self.qplus

        qplusn=self.tpt.forward_committor
        self.assertTrue(np.allclose(qplusn, qplus))

        qplusn=self.tpt_fast.forward_committor
        self.assertTrue(np.allclose(qplusn, qplus))

    def test_stationary_distribution(self):
        mu=self.mu
        
        mun=self.tpt.stationary_distribution
        self.assertTrue(np.allclose(mun, mu))

        mun=self.tpt_fast.stationary_distribution
        self.assertTrue(np.allclose(mun, mu))


class TestTptFunctionsDense(unittest.TestCase):
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

        import pyemma.msm.analysis.dense.birth_death_chain
        self.bdc=pyemma.msm.analysis.dense.birth_death_chain.BirthDeathChain(q, p)
        self.T=self.bdc.transition_matrix()    

        self.mu=self.bdc.stationary_distribution()
        self.qminus=self.bdc.committor_backward(self.a, self.b)
        self.qplus=self.bdc.committor_forward(self.a, self.b)
        
        # present results
        self.fluxn = flux.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=False)
        self.netfluxn = flux.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=True)
        self.totalfluxn=flux.total_flux(self.netfluxn, self.A)
        self.raten=flux.rate(self.totalfluxn, self.mu, self.qminus)
    
    def test_tpt_flux(self):
        flux=self.bdc.flux(self.a, self.b)        
        self.assertTrue(np.allclose(self.fluxn, flux))

    def test_tpt_netflux(self):
        netflux=self.bdc.netflux(self.a, self.b)
        self.assertTrue(np.allclose(self.netfluxn, netflux))

    def test_tpt_totalflux(self):
        totalflux=self.bdc.totalflux(self.a, self.b)
        self.assertTrue(np.allclose(self.totalfluxn, totalflux))

    def test_tpt_rate(self):
        rate=self.bdc.rate(self.a, self.b)
        self.assertTrue(np.allclose(self.raten, rate))


################################################################################
# Sparse
################################################################################

class TestTPTSparse(unittest.TestCase):
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
            
        import pyemma.msm.analysis.sparse.birth_death_chain
        self.bdc=pyemma.msm.analysis.sparse.birth_death_chain.BirthDeathChain(q, p)
        T_dense=self.bdc.transition_matrix()
        T_sparse=csr_matrix(T_dense)
        self.T=T_sparse

        """Compute mu, qminus, qplus in constructor"""
        self.tpt=flux.tpt(self.T, self.A, self.B)

        """Use precomputed mu, qminus, qplus"""        
        self.mu=self.bdc.stationary_distribution()
        self.qminus=self.bdc.committor_backward(self.a, self.b)
        self.qplus=self.bdc.committor_forward(self.a, self.b)
        self.tpt_fast=flux.tpt(self.T, self.A, self.B,\
                              mu=self.mu, qminus=self.qminus,\
                              qplus=self.qplus)

    def test_flux(self):
        flux=self.bdc.flux(self.a, self.b)        

        fluxn=self.tpt.gross_flux
        self.assertTrue(np.allclose(fluxn.toarray(), flux))

        fluxn=self.tpt_fast.gross_flux
        self.assertTrue(np.allclose(fluxn.toarray(), flux))

    def test_netflux(self):
        netflux=self.bdc.netflux(self.a, self.b)
        
        netfluxn=self.tpt.net_flux
        self.assertTrue(np.allclose(netfluxn.toarray(), netflux))

        netfluxn=self.tpt_fast.net_flux
        self.assertTrue(np.allclose(netfluxn.toarray(), netflux))        

    def test_totalflux(self):
        F=self.bdc.totalflux(self.a, self.b)

        Fn=self.tpt.total_flux
        self.assertTrue(np.allclose(Fn, F))

        Fn=self.tpt_fast.total_flux
        self.assertTrue(np.allclose(Fn, F))

    def test_rate(self):
        k=self.bdc.rate(self.a, self.b)
        
        kn=self.tpt.rate
        self.assertTrue(np.allclose(kn, k))

        kn=self.tpt_fast.rate
        self.assertTrue(np.allclose(kn, k))

    def test_backward_committor(self):
        qminus=self.qminus

        qminusn=self.tpt.backward_committor
        self.assertTrue(np.allclose(qminusn, qminus))

        qminusn=self.tpt_fast.backward_committor
        self.assertTrue(np.allclose(qminusn, qminus))

    def test_forward_committor(self):
        qplus=self.qplus

        qplusn=self.tpt.forward_committor
        self.assertTrue(np.allclose(qplusn, qplus))

        qplusn=self.tpt_fast.forward_committor
        self.assertTrue(np.allclose(qplusn, qplus))

    def test_stationary_distribution(self):
        mu=self.mu
        
        mun=self.tpt.stationary_distribution
        self.assertTrue(np.allclose(mun, mu))

        mun=self.tpt_fast.stationary_distribution
        self.assertTrue(np.allclose(mun, mu))

class TestTptFunctionsSparse(unittest.TestCase):
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
            
        import pyemma.msm.analysis.sparse.birth_death_chain
        self.bdc=pyemma.msm.analysis.sparse.birth_death_chain.BirthDeathChain(q, p)
        T_dense=self.bdc.transition_matrix()
        T_sparse=csr_matrix(T_dense)
        self.T=T_sparse

        self.mu=self.bdc.stationary_distribution()
        self.qminus=self.bdc.committor_backward(self.a, self.b)
        self.qplus=self.bdc.committor_forward(self.a, self.b)

        # present results
        self.fluxn = flux.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=False)
        self.netfluxn = flux.flux_matrix(self.T, self.mu, self.qminus, self.qplus, netflux=True)
        self.totalfluxn=flux.total_flux(self.netfluxn, self.A)
        self.raten=flux.rate(self.totalfluxn, self.mu, self.qminus)
    
    def test_tpt_flux(self):
        flux=self.bdc.flux(self.a, self.b)
        self.assertTrue(np.allclose(self.fluxn.toarray(), flux))

    def test_tpt_netflux(self):
        netflux=self.bdc.netflux(self.a, self.b)
        self.assertTrue(np.allclose(self.netfluxn.toarray(), netflux))

    def test_tpt_totalflux(self):
        totalflux=self.bdc.totalflux(self.a, self.b)
        self.assertTrue(np.allclose(self.totalfluxn, totalflux))

    def test_tpt_rate(self):
        rate=self.bdc.rate(self.a, self.b)
        self.assertTrue(np.allclose(self.raten, rate))


if __name__ == "__main__":
    unittest.main()
