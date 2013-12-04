'''
Created on 30.11.2013

@author: jan-hendrikprinz
'''

import pcca_impl
import decomposition

import numpy

def pcca(T, n):
    
    eigenvalues,left_eigenvectors,right_eigenvectors = decomposition.rdl_decomposition(T, n)
    
    # create initial solution that works
    
    # This creates a linear combination of the eigenfunctions with growing complexity
    # fill_matrix ensures that the rest of the matrix is created s.t. the sum of all memberships is one
    # and no entry is negative !
    # Don't know if the initial one makes sense, but I thought this is easy to setup and it gets a valid membership
    # Could be better to use the old PCCA solution. Therefore the function cluster_by_isa exists in the implementation
    
    lowertriangular_indices = numpy.triu_indices(n - 1)
    rot_crop_matrix = numpy.zeros(n - 1,n -1)
    rot_crop_matrix[lowertriangular_indices] = 1    
    
    rot_matrix = pcca_impl.fill_matrix(rot_crop_matrix, right_eigenvectors)

    # use the predefined matrix as initial guess for the optimizer
    
    rot_matrix = pcca_impl.opt_soft(right_eigenvectors, rot_matrix, n)
    
    memberships = numpy.dot(right_eigenvectors[:,1:], rot_crop_matrix)
    
    return memberships