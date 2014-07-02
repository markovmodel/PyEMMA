r"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import numpy as np

from decomposition import stationary_distribution_from_backward_iteration as statdist
from committor import forward_committor, backward_committor

class TPT(object):
    def __init__(self, T, A, B, mu=None, qminus=None, qplus=None, name_A=None, name_B=None):
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
        mu : (M,) ndarray (optional)
            Stationary vector
        qminus : (M,) ndarray (optional)
            Backward committor for A->B reaction
        qplus : (M,) ndarray (optional)
            Forward committor for A-> B reaction
        name_A : string
            optional name for set A
        name_b : string
            optional name for set B
         
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
        if name_A:
            self.name_A = name_A
        if name_B:
            self.name_B = name_B

        """Precompute quantities required for TPT-analysis (if necessary)"""
        if mu is None:
            mu=statdist(T)
        if qminus is None:
            qminus=backward_committor(T, A, B)               
        if qplus is None:
            qplus=forward_committor(T, A, B)
        self.mu=mu
        self.qminus=qminus
        self.qplus=qplus

        """Compute TPT-quantities"""
        self.flux=self._flux(self.T, self.mu, self.qminus, self.qplus)
        self.netflux=self._netflux(self.flux)
        self.totalflux=self._totalflux(self.flux, self.A)
        self.rate=self._rate(self.totalflux, self.mu, self.qminus)
          
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

    def get_backward_committor(self):
        r"""Backward committor for A to B reaction.

        Returns
        -------
        qminus : (M,) ndarray
            Backward committor

        """
        return self.qminus

    def get_forward_committor(self):
        r"""Forward committor for A to B reaction.

        Returns
        -------
        qplus : (M,) ndarray
            Forward committor

        """
        return self.qplus

    def get_stationary_distribution(self):
        r"""Stationary distribution. 

        Returns
        -------
        mu : (M,) ndarray
            Stationary distribution
            
        """
        return self.mu
    
    def __getattr__(self, name):
        """
        if user have not given a name for set A/B, the string repr of the set
        is returned. Note that this method is only called for not defined attributes.
        """
        if name == 'name_A':
            return repr(self.A)
        if name == 'name_B':
            return repr(self.B)
        else:
            # Default behavior
            raise AttributeError("no attribute named '%s'" % name)
    

def tpt_flux(T, A, B, mu=None, qminus=None, qplus=None):
    r"""Flux network for the reaction A -> B.
    
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    mu : (M,) ndarray (optional)
        Stationary vector
    qminus : (M,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (M,) ndarray (optional)
        Forward committor for A-> B reaction
        
    Returns
    -------
    flux : (M, M) ndarray
        Matrix of flux values between pairs of states.
        
    Notes
    -----
    Computation of the flux network relies on transition path theory
    (TPT). The central object used in transition path theory is the
    forward and backward comittor function.
    
    See also
    --------
    committor.forward_committor, committor.backward_committor
    
    """    
    tpt=TPT(T, A, B, mu=mu, qminus=qminus, qplus=qplus)
    return tpt.get_flux()

def tpt_netflux(T, A, B, mu=None, qminus=None, qplus=None):
    r"""Netflux network for the reaction A -> B.
    
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    mu : (M,) ndarray (optional)
        Stationary vector
    qminus : (M,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (M,) ndarray (optional)
        Forward committor for A-> B reaction

    Returns
    -------
    netflux : (M, M) ndarray
        Matrix of netflux values between pairs of states.

    Notes
    -----
    Computation of the netflux network relies on transition path theory
    (TPT). The central object used in transition path theory is the
    forward and backward comittor function.

    See also
    --------
    committor.forward_committor, committor.backward_committor

    """    
    tpt=TPT(T, A, B, mu=mu, qminus=qminus, qplus=qplus)
    return tpt.get_netflux()

def tpt_totalflux(T, A, B, mu=None, qminus=None, qplus=None):
    r"""Total flux for the reaction A -> B.
    
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    mu : (M,) ndarray (optional)
        Stationary vector
    qminus : (M,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (M,) ndarray (optional)
        Forward committor for A-> B reaction

    Returns
    -------
    F : float
        The total flux between reactant and product

    Notes
    -----
    Computation of the total flux network relies on transition path
    theory (TPT). The central object used in transition path theory is
    the forward and backward comittor function.

    See also
    --------
    committor.forward_committor, committor.backward_committor

    """    
    tpt=TPT(T, A, B, mu=mu, qminus=qminus, qplus=qplus)
    return tpt.get_totalflux()

def tpt_rate(T, A, B, mu=None, qminus=None, qplus=None):
    r"""Rate of the reaction A -> B.
    
    Parameters
    ----------
    T : (M, M) ndarray
        Transition matrix
    A : array_like
        List of integer state labels for set A
    B : array_like
        List of integer state labels for set B
    mu : (M,) ndarray (optional)
        Stationary vector
    qminus : (M,) ndarray (optional)
        Backward committor for A->B reaction
    qplus : (M,) ndarray (optional)
        Forward committor for A-> B reaction

    Returns
    -------
    kAB : float
        The reaction rate (per time step of the Markov chain)

    Notes
    -----
    Computation of the rate relies on transition path theory
    (TPT). The central object used in transition path theory is the
    forward and backward comittor function.

    See also
    --------
    committor.forward_committor, committor.backward_committor

    """    
    tpt=TPT(T, A, B, mu=mu, qminus=qminus, qplus=qplus)
    return tpt.get_rate()
