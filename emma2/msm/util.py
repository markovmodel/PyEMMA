'''
Created on 24.09.2013

@author: marscher
'''
import numpy as np


def isProbabilisticMatrix(T):
    """
    returns whether T is a probabilistic matrix.
    Only implemented for NumPy array types
    """
    if not isinstance(T, np.ndarray):
        raise NotImplemented("only impled for NumPy ndarray type")
    
    # check that matrix is quadratic
    if T.shape[0] != T.shape[1]:
        return False
    
    r = False
    # check that all values lies between [0.0, 1.0]
    r = (T >= 0.0).all() and (T <= 1).all()
    # check row sums
    for row in xrange(0, T.shape[1]):
        diff =  abs(T[row].sum() - 1)
        if diff >= 1e-6:
            print "Sum of row %i is greater 1.0. Sum is: %d" % (row, diff)
            r = False
            break
    
    return r
