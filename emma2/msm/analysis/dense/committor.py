'''
Created on 08.11.2013

@author: marscher
'''
import numpy

def forward_committor(T, A, B):
    """ 
    calculate forward committor from A to B given transition matrix T.
    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    A : array like
        List of integer state labels for set A
    B : array like
        List of integer state labels for set B
        
    Returns
    -------
    x : ndarray, shape=(n, )
    Committor vector.
    """
    n = len(T)
    set_X = numpy.arange(n)
    set_A = numpy.unique(A)
    set_B = numpy.unique(B)
    set_AB = numpy.intersect1d(set_A, set_B, True)
    notAB = numpy.setdiff1d(set_X, set_AB, True)
    m = len(notAB)

    K = T - numpy.diag(numpy.ones((n)))

    U = K[numpy.ix_(notAB, notAB)]
    v = numpy.zeros(m)
    v[:] = v[:] - K[notAB[:], B[:]]

    qI = numpy.linalg.solve(U, v)

    q_forward = numpy.zeros(n)
    q_forward[set_B] = 1
    q_forward[notAB[:]] = qI[:]

    return q_forward

def backward_committor(T, A, B):
    """ 
    calculate backward committor from A to B given transition matrix T.
    Parameters
    ----------
    T : numpy.ndarray shape = (n, n)
        Transition matrix
    A : array like
        List of integer state labels for set A
    B : array like
        List of integer state labels for set B
        
    Returns
    -------
    x : ndarray, shape=(n, )
    Committor vector.
    """
    from ..api import statdist
    n = len(T)
    set_X = numpy.arange(n)
    set_A = numpy.unique(A)
    set_B = numpy.unique(B)
    set_AB = numpy.intersect1d(set_A, set_B, True)
    notAB = numpy.setdiff1d(set_X, set_AB, True)
    m = len(notAB)
    
    eq = statdist(T)
        
    Tback = numpy.transpose(T)
    Tback = numpy.dot(numpy.diag(1.0 / eq), Tback)
    Tback = numpy.dot(Tback, numpy.diag(eq))
        
    K = Tback - numpy.diag(numpy.ones((n)))

    U = K[numpy.ix_(notAB, notAB)]
    v = numpy.zeros(m)
    v[:] = v[:] - K[notAB[:], B[:]]

    qI = numpy.linalg.solve(U, v)

    q_backward = numpy.zeros(n)
    q_backward[set_B] = 1
    q_backward[notAB[:]] = qI[:]

    return q_backward