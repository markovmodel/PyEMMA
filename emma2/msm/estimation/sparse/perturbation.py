'''
Created on 25.11.2013

@author: jan-hendrikprinz
'''
from transition_matrix import tmatrix_cov

import numpy

#TODO: maybe use symmetry in the covariance matrices to reduce computation by one order
def error_perturbation(C, sensitivity):
    error = 0.0;
    
    n = len(C)
    
    for k in range(0,n):
        cov = tmatrix_cov(C, k)
        error += numpy.dot(numpy.dot(sensitivity[k],cov),sensitivity[k])

    return error