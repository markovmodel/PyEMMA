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
    set_X = set(range(n))
    set_A = set(A)
    set_B = set(B)
    set_AB = set_A | set_B
    notAB = list(set_X - set_AB)
    m = len(notAB)

    K = T - numpy.diag(numpy.ones((n)))

    U = K[numpy.ix_(notAB, notAB)]
    v = numpy.zeros((m))
    for i in range(0, m):
        for k in range(0, len(set_B)):
            v[i] = v[i] - K[notAB[i], B[k]]

    qI = numpy.linalg.solve(U, v)

    q_forward = numpy.zeros((n))
    for i in set_A:
        q_forward[i] = 0
    for i in set_B:
        q_forward[i] = 1
    for i in range(len(notAB)):
        q_forward[notAB[i]] = qI[i]

    return q_forward