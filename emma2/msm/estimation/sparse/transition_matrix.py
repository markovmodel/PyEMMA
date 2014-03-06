"""This module implements the transition matrix functionality"""

import numpy
import scipy.sparse

def transition_matrix_non_reversible(C):
    """implementation of transition_matrix"""
    if not scipy.sparse.issparse(C):
        C = scipy.sparse.csr_matrix(C)
    rowsum = C.tocsr().sum(axis=1)
    # catch div by zero
    if(numpy.min(rowsum) == 0.0):
        raise ValueError("matrix C contains rows with sum zero.")
    rowsum = numpy.array(1. / rowsum).flatten()
    norm = scipy.sparse.diags(rowsum, 0)
    return norm * C


def tmatrix_cov(C, row=None):
    if row is not None:
        matrix = dirichlet_covariance(C[row])
        return matrix
    else:
        size = C.shape[1]
        tensor = numpy.zeros((size,size,size))
        for i in range(0,size):
            tensor[i] = dirichlet_covariance(C[i]);
        return tensor


def error_perturbation(C, sensitivity):
    error = 0.0;
    
    n = len(C)
    
    for k in range(0,n):
        cov = tmatrix_cov(C, k)
        error += numpy.dot(numpy.dot(sensitivity[k],cov),sensitivity[k])
    return error


def dirichlet_covariance(c):
    """Returns a matrix of covariances between all elements of the dirichlet distribution parametrized by the vector c
        
    Parameters
    ----------
    c : numpy.ndarray
    
    Returns
    -------
    cov : numpy.array
    
    """
    
    cTotal = numpy.sum(c)
    
    cNorm = c / cTotal
    
    mOff = numpy.outer(cNorm,cNorm)
    mDiag = numpy.diagflat(cNorm)
    
    norm = 1.0 / (cTotal + 1.0)
    
    cov = norm * (mDiag - mOff)  
    
    return cov