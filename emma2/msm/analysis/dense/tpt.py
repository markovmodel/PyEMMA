r"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import numpy as np


def grossflux(T, pi, qminus, qplus):
    r"""Compute the TPT gross flux network for the reaction A-->B.
    
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
    
    Notes
    -----
    Computation of the flux network relies on transition path theory
    (TPT). The central object used in transition path theory is the
    forward and backward comittor function.
    
    See also
    --------
    committor.forward_committor, committor.backward_committor
    
    
    """
    flux=pi[:,np.newaxis]*qminus[:,np.newaxis]*T*\
        qplus[np.newaxis,:]
    ind=np.diag_indices(T.shape[0])
    """Remove self fluxes f_ii"""
    flux[ind]=0.0
    return flux


def netflux(flux):
    r"""Compute the netflux from the gross flux.
    
        f_ij^{+}=max{0, f_ij-f_ji}
        for all pairs i,j
    
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


def totalflux(flux, A):
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



def rate(F, pi, qminus):
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

