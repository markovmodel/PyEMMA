r"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin.trendelkampschroer@gmail.com>

"""
import numpy as np

from decomposition import stationary_distribution_from_backward_iteration as statdist
from committor import forward_committor, backward_committor

class TPT:
    def __init__(self, T, A, B):
        r""" A multi-purpose TPT-object.

        The TPT-object provides methods for Transition Path analysis 
        of transition matrices.

        Parameters
        ----------
        T : (M, M) ndarray
            Transition matrix
        A : array_like
            List of integer state labels for set A
        B : array_like
            List of integer state labels for set B

        Notes
        -----
        The central object used in transition path theory is
        the forward and backward comittor function.

        See also
        --------
        committor.forward_committor, committor.backward_committor

        """    
        self.T=T
        self.A=A
        self.B=B

        """Precompute quantities required for TPT-analysis"""
        self.pi=statdist(T)
        self.qplus=forward_committor(T, A, B)
        self.qminus=backward_committor(T, A, B)   

        """Compute TPT-quantities"""
        self.flux=self._flux(self.T, self.pi, self.qminus, self.qplus)
        self.netflux=self._netflux(self.flux)
        self.totalflux=self._totalflux(self.flux, self.A)
        self.rate=self._rate(self.totalflux, self.pi, self.qminus)
          
    def _flux(self, T, pi, qminus, qplus):
        r"""Compute the flux.

        Parameters
        ----------
        T : (M, M) ndarray
            transition matrix
        pi : (M,) ndarray
            Stationary distribution corresponding to T
        qminus : (M,) ndarray
            Backward comittor
        qplus : (M,) ndarray
            Forward committor

        Returns
        -------
        flux : (M, M) ndarray
            Matrix of flux values between pairs of states.

        """
        flux=pi[:,np.newaxis]*qminus[:,np.newaxis]*T*\
            qplus[np.newaxis,:]
        ind=np.diag_indices(T.shape[0])
        """Remove self fluxes f_ii"""
        flux[ind]=0.0
        return flux

    def _netflux(self, flux):
        r"""Compute the netflux.

            f_ij^{+}=max{0, f_ij-f_ji}

        Parameters
        ----------
        flux : (M, M) ndarray
            Matrix of flux values between pairs of states.

        Returns
        -------
        netflux : (M, M) ndarray
            Matrix of netflux values between pairs of states.

        """
        netflux=flux-np.transpose(flux)
        """Set negative fluxes to zero"""
        ind=(netflux<0.0)
        netflux[ind]=0.0       
        return netflux       

    def _totalflux(self, flux, A):
        r"""Compute the total flux between reactant and product.

        Parameters
        ----------
        flux : (M, M) ndarray
            Matrix of flux values between pairs of states.
        A : array_like
            List of integer state labels for set A (reactant)

        Returns
        -------
        F : float
            The total flux between reactant and product

        """
        X=set(np.arange(flux.shape[0])) # total state space
        A=set(A)
        notA=X.difference(A)
        F=(flux[list(A),:])[:,list(notA)].sum()
        # F=(flux[list(A), list(notA)]).sum()
        return F        

    def _rate(self, F, pi, qminus):
        r"""Transition rate for reaction A to B.

        Parameters
        ----------
        F : float
            The total flux between reactant and product
        pi : (M,) ndarray
            Stationary distribution
        qminus : (M,) ndarray
            Backward comittor

        Returns
        -------
        kAB : float
            The reaction rate (per time step of the
            Markov chain)

        """
        kAB=F/(pi*qminus).sum()
        return kAB        

    def get_flux(self):
        r"""The flux network for the reaction.

        Returns
        -------
        flux : (M, M) ndarray
            Matrix of flux values between pairs of states.

        """
        return self.flux

    def get_netflux(self):
        r"""The netflux network for the reaction.

        Returns
        -------
        netflux : (M, M) ndarray
            Matrix of netflux values between pairs of states.

        """
        return self.netflux

    def get_totalflux(self):
        r"""Total flux between reactant and product.

        Returns
        -------
        F : float
            The total flux between reactant and product

        """
        return self.totalflux

    def get_rate(self):
        r"""Transition rate for reaction A to B.

        Returns
        -------
        kAB : float
            The reaction rate (per time step of the
            Markov chain)
            
        """
        return self.rate
    
