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
        matrix = dirichlet_covariance(C[row]+1)
        return matrix
    else:
        size = C.shape[1]
        tensor = numpy.empty((size, size, size))
        i = numpy.arange(size)
        tensor[i] = dirichlet_covariance(C[i]+1);
        return tensor


def error_perturbation(C, sensitivity):
    error = 0.0;
    
    n = C.shape[0]
    
    for k in range(0,n):
        cov = tmatrix_cov(C, k)
        error += numpy.dot(numpy.dot(sensitivity[k],cov),sensitivity[k])
    return error


#TODO: Check for integer array type and convert if necessary
def dirichlet_covariance(u):
    """Returns a matrix of covariances between all elements of the dirichlet distribution parametrized by the vector u
        
    Parameters
    ----------
    u : numpy.ndarray
        Dirichlet parameters (note that these are counts + 1)
    
    Returns
    -------
    cov : numpy.array
    
    """
    
    cTotal = numpy.sum(u)
    cNorm = (1.0 * u) / cTotal
    
    mOff = numpy.outer(cNorm,cNorm)
    mDiag = numpy.diagflat(cNorm)
    
    norm = 1.0 / (cTotal + 1.0)
    cov = norm * (mDiag - mOff)  
    
    return cov