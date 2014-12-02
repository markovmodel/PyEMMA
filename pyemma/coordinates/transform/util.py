'''
Created on Jan 5, 2014

@author: noe
'''

import numpy as np

def project(x, T, ndim = None):
    """
    Projects vector x onto the coordinate system T

    Performs a linear projection of vector x to a coordinate system
    given by the column vectors in T. This projection is used in linear
    dimension reduction methods such as PCA and TICA, where T are the
    respective eigenvector matrices.

    Parameters
    ----------
    x : ndarray (n)
        coordinate vector to be projected
    T : ndarray (n x n)
        coordinate system with vectors in its columns
    ndim = None : int
        Dimension of the output vector. Only the first ndim vectors of T
        will be used for projection. When set to None (default), no dimension
        reduction will be made, and the output vector has size n. 

    Returns
    -------
    Projected coordinate vector

    """
    if (ndim is None):
        return np.dot(x.T, T)
    else:
        return np.dot(x.T, T[:, 0: ndim])