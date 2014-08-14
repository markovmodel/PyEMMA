r"""This module contains function for the Transition Path Theory (TPT)
analysis of Markov models.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import numpy as np

from scipy.sparse import diags, coo_matrix


def remove_negative_entries(A):
    r"""Remove all negative entries from sparse matrix.

        Aplus=max(0, A)

    Parameters
    ----------
    A : (M, M) scipy.sparse matrix
        Input matrix

    Returns
    -------
    Aplus : (M, M) scipy.sparse matrix
        Input matrix with negative entries set to zero.

    """
    A=A.tocoo()

    data=A.data
    row=A.row
    col=A.col

    """Positive entries"""
    pos=data>0.0
    
    datap=data[pos]
    rowp=row[pos]
    colp=col[pos]

    Aplus=coo_matrix((datap, (rowp, colp)), shape=A.shape)
    return Aplus


def grossflux(T, pi, qminus, qplus):
    r"""Compute the flux.
    
    Parameters
    ----------
    T : (M, M) scipy.sparse matrix
        Transition matrix
    pi : (M,) ndarray
        Stationary distribution corresponding to T
    qminus : (M,) ndarray
        Backward comittor
    qplus : (M,) ndarray
        Forward committor
    
    Returns
    -------
    flux : (M, M) scipy.sparse matrix
        Matrix of flux values between pairs of states.
    
    """
    D1=diags((pi*qminus,), (0,))
    D2=diags((qplus,), (0,))
    
    flux=D1.dot(T.dot(D2))
    
    """Remove self-fluxes"""
    flux=flux-diags(flux.diagonal(), 0)        
    return flux


def netflux(flux):
    r"""Compute the netflux.
    
    f_ij^{+}=max{0, f_ij-f_ji}
    
    Parameters
    ----------
    flux : (M, M) scipy.sparse matrix
        Matrix of flux values between pairs of states.
    
    Returns
    -------
    netflux : (M, M) scipy.sparse matrix
        Matrix of netflux values between pairs of states.
    
    """
    netflux=flux-flux.T
    
    """Set negative entries to zero"""
    netflux=remove_negative_entries(netflux)
    return netflux


def totalflux(flux, A):
    r"""Compute the total flux between reactant and product.
    
    Parameters
    ----------
    flux : (M, M) scipy.sparse matrix
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
    
    """Extract rows corresponding to A"""
    W=flux.tocsr()
    W=W[list(A), :]
    """Extract columns corresonding to X\A"""
    W=W.tocsc()
    W=W[:,list(notA)]
    
    F=W.sum()
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


