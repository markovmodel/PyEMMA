# -*- coding: utf-8 -*-
###############################################################################################################################################
#
# Eigenvalues and Eigenvectors
#
###############################################################################################################################################


def timescales(T, tau):
    """
        T: transition matrix
        tau: lag time
    """

def statdist(T):
    """
        compute the stationary distribution of T
    """
    pass

def statdist_sensitivity(T):
    """
        compute the sensitivity matrix of the stationary distribution of T
    """    

def eigenvalues(T, n):
    """
        computes the first n eigenvalues
    """
    
def eigenvalue_sentivity(T, eigenvalue):
    """
        computes the sensitivity of the specified eigenvalue
    """

def left_eigenvectors(T, n):
    """
        computes the first left eigenvectors
    """

def left_eigenvector_sensitivity(T, index):
    """
        computes the sensitivity of the given left eigenvector
    """

def right_eigenvectors(T, n):
    """
        computes the first right eigenvectors
    """

def right_eigenvector_sensitivity(T, index):
    """
        computes the sensitivity of the given right eigenvector
    """



###############################################################################################################################################
#
# PCCA
#
###############################################################################################################################################

def pcca(T, n):
    """
        returns a PCCA object
        T: transition matrix
        n: number of metastable processes
    """

###############################################################################################################################################
#
# Committor and TPT function
#
###############################################################################################################################################

def committor_forward(T, A, B):
    """
        T: transition matrix
        A: set A
        B: set B
    """

def committor_forward_sensitivity(T, A, B):
    """
        T: transition matrix
        A: set A
        B: set B
    """

def committor_backward(T, A, B):
    """
        T: transition matrix
        A: set A
        B: set B
    """

def committor_backward_sensitivity(T, A, B):
    """
        T: transition matrix
        A: set A
        B: set B
    """

def tpt(T, A, B):
    """
        returns a TPT object
    """


###############################################################################################################################################
#
# Expectation and correlation functions
#
###############################################################################################################################################


def expectation(T, a):
    """
        computes the expectation value of a
    """

def expectation_sensitivity(T, a):    
    """
        computes the sensitivity of the expectation value of a
    """

def correlation(T, a, b, lagtimes):
    """
        computes the time correlation function of a and b
    """

def correlation_sensitivity(T, a, b, lagtimes):    
    """
        computes the sensitivity of the time correlation function of a and b
    """

def relaxation(T, a, p0, lagtimes):
    """
        computes the time relaxation function of a starting from p0
    """

def relaxation_sensitivity(T, a, p0, lagtimes):    
    """
        computes the sensitivity of the time relaxation function of a starting from p0
    """
