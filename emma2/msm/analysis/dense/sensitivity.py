'''
Created on 22.11.2013

@author: jan-hendrikprinz
'''

import numpy

# TODO:make faster. So far not effectively programmed
def forward_committor_sensitivity(T, A, B, index):
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
    index : entry of the committor for which the sensitivity is to be computed
        
    Returns
    -------
    x : ndarray, shape=(n, n)
    Sensitivity matrix for entry index around transition matrix T. Reversibility is not assumed.
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

    Uinv = numpy.linalg.inv(U)
    Siab = numpy.zeros((n,n))
    
    for i in range(0, m):
        for a in range(0, m):
            Siab[notAB[i],notAB[a]] = - Uinv[a,i] * q_forward[index]

    return Siab

def eigenvalue_sensitivity(T, k):
        
    eValues, rightEigenvectors = numpy.linalg.eig(T)
    leftEigenvectors = numpy.linalg.inv(rightEigenvectors)    
    
    perm = numpy.argsort(eValues)[::-1]

    rightEigenvectors=rightEigenvectors[perm]
    leftEigenvectors=leftEigenvectors[perm]
        
    sensitivity = numpy.outer(leftEigenvectors[k], rightEigenvectors[k])
    
    return sensitivity

def eigenvector_sensitivity(T, k, j, right=True):
    
    n = len(T)
    
    eValues, rightEigenvectors = numpy.linalg.eig(T)
    leftEigenvectors = numpy.linalg.inv(rightEigenvectors)    
    
    perm = numpy.argsort(eValues)[::-1]

    rightEigenvectors=rightEigenvectors[perm]
    leftEigenvectors=leftEigenvectors[perm]
    
    matA = T - numpy.diag(numpy.ones((n)))
    
    matAInv = numpy.linalg.pinv(matA, 10.^-12)
            
    sensitivity = numpy.outer(leftEigenvectors[k], rightEigenvectors[k])

    return sensitivity