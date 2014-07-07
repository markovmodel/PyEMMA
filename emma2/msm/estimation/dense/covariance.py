r"""This module implements the transition matrix covariance function

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np

def tmatrix_cov(C, row=None):
    r"""Covariance tensor for the non-reversible transition matrix ensemble

    Parameters
    ----------
    C : (M, M) ndarray
        Count matrix
    row : int (optional)
        If row is given return covariance matrix for specified row only

    Returns
    -------
    cov : (M, M, M) ndarray
        Covariance tensor

    """

    if row is None:
        alpha=C+1.0 #Dirichlet parameters
        alpha0=alpha.sum(axis=1) #Sum of paramters (per row)
    
        norm=alpha0**2*(alpha0+1.0)

        """Non-normalized covariance tensor"""
        Z=-alpha[:,:,np.newaxis]*alpha[:,np.newaxis,:]

        """Correct-diagonal"""
        ind=np.diag_indices(C.shape[0])
        Z[:, ind[0], ind[1]]+=alpha0[:,np.newaxis]*alpha

        """Covariance matrix"""
        cov=Z/norm[:,np.newaxis,np.newaxis]

        return cov       
    
    else:
        alpha=C[row, :]+1.0
        return dirichlet_covariance(alpha)

def dirichlet_covariance(alpha):
    r"""Covariance matrix for Dirichlet distribution.

    Parameters
    ----------
    alpha : (M, ) ndarray
        Parameters of Dirichlet distribution
    
    Returns
    -------
    cov : (M, M) ndarray
        Covariance matrix
        
    """
    alpha0=alpha.sum()
    norm=alpha0**2*(alpha0+1.0)

    """Non normalized covariance"""
    Z=-alpha[:,np.newaxis]*alpha[np.newaxis,:]

    """Correct diagonal"""
    ind=np.diag_indices(Z.shape[0])
    Z[ind]+=alpha0*alpha

    """Covariance matrix"""
    cov=Z/norm
    
    return cov

def error_perturbation(C, sensitivity):
    error = 0.0;
    
    n = C.shape[0]
    
    for k in range(0,n):
        cov = tmatrix_cov(C, k)
        error += numpy.dot(numpy.dot(sensitivity[k],cov),sensitivity[k])
    return error

