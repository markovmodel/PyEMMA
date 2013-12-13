'''
Created on 22.11.2013

@author: Jan-Hendrik Prinz
'''

import numpy
from emma2.autobuilder.emma_msm_mockup import stationary_distribution

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
        
    target = numpy.eye(1,n,index)
    target = target[0,notAB]

    UinvVec = numpy.linalg.solve(numpy.transpose(U), target)
    Siab = numpy.zeros((n,n))
        
    for i in range(0, m):
        Siab[notAB[i]] = - UinvVec[i] * q_forward

    return Siab

def backward_committor_sensitivity(T, A, B, index):
    """ 
    calculate the sensitivity of index of the backward committor from A to B given transition matrix T.
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
    
    # This is really ugly to compute. The problem is, that changes in T induce changes in
    # the stationary distribution and so we need to add this influence, too
    
    n = len(T)
    
    trT = numpy.transpose(T)
    
    one = numpy.ones(n)
    eq = stationary_distribution(T)
    
    mEQ = numpy.diag(eq)
    mIEQ = numpy.diag(1.0 / eq)
    mSEQ = numpy.diag(1.0 / eq / eq)
    
    backT = numpy.dot(mIEQ, numpy.dot( trT, mEQ))
    
    qMat = forward_committor_sensitivity(backT, A, B, index)
    
    matA = trT - numpy.identity(n)
    matA = numpy.concatenate((matA, [one]))
    
    phiM = numpy.linalg.pinv(matA)
    
    phiM = phiM[:,0:n]
    
    trQMat = numpy.transpose(qMat)
    
    d1 = numpy.dot( mSEQ, numpy.diagonal(numpy.dot( numpy.dot(trT, mEQ), trQMat), 0) )
    d2 = numpy.diagonal(numpy.dot( numpy.dot(trQMat, mIEQ), trT), 0)
        
    psi1 = numpy.dot(d1, phiM)
    psi2 = numpy.dot(-d2, phiM)
    
    v1 = psi1 - one * numpy.dot(psi1, eq)
    v3 = psi2 - one * numpy.dot(psi2, eq)
    
    part1 = numpy.outer(eq, v1)
    part2 = numpy.dot( numpy.dot(mEQ, trQMat), mIEQ)
    part3 = numpy.outer(eq, v3)
    
    sensitivity = part1 + part2 + part3
    
    return sensitivity

def eigenvalue_sensitivity(T, k):
        
    eValues, rightEigenvectors = numpy.linalg.eig(T)
    leftEigenvectors = numpy.linalg.inv(rightEigenvectors)    
    
    perm = numpy.argsort(eValues)[::-1]

    rightEigenvectors=rightEigenvectors[:,perm]
    leftEigenvectors=leftEigenvectors[perm]
    
    sensitivity = numpy.outer(leftEigenvectors[k], rightEigenvectors[:,k])
    
    return sensitivity

# TODO: The eigenvector sensitivity depends on the normalization, e.g. l^T r = 1 or norm(r) = 1
# Should we fix that or add another option. Also the sensitivity depends on the initial eigenvectors
# Now everything is set to use norm(v) = 1
def eigenvector_sensitivity(T, k, j, right=True):
    
    n = len(T)
    
    if not right:    
        T = numpy.transpose(T)
    
    eValues, rightEigenvectors = numpy.linalg.eig(T)
    leftEigenvectors = numpy.linalg.inv(rightEigenvectors)        
    perm = numpy.argsort(eValues)[::-1]

    eValues = eValues[perm]
    rightEigenvectors=rightEigenvectors[:,perm]
    leftEigenvectors=leftEigenvectors[perm]
        
    rEV = rightEigenvectors[:,k]
    lEV = leftEigenvectors[k]
    eVal = eValues[k]
    
    vecA = numpy.zeros(n)
    vecA[j] = 1.0
           
    matA = T - eVal * numpy.identity(n)
        # Use here rEV as additional condition, means that we assume the vector to be
        # orthogonal to rEV
    matA = numpy.concatenate((matA, [rEV]))
                
    phi = numpy.linalg.lstsq(numpy.transpose(matA), vecA)    
        
    phi = numpy.delete(phi[0], -1)
                
    sensitivity = -numpy.outer(phi,rEV) + numpy.dot(phi,rEV) * numpy.outer(lEV, rEV) 
    
    if not right:
        sensitivity = numpy.transpose(sensitivity)          
        
    return sensitivity

def stationary_distribution_sensitivity(T, j):
        
    n = len(T)
        
    lEV = numpy.ones(n)
    rEV = stationary_distribution(T)
    eVal = 1.0
    
    T = numpy.transpose(T)
    
    vecA = numpy.zeros(n)
    vecA[j] = 1.0
               
    matA = T - eVal * numpy.identity(n)
    # normalize s.t. sum is one using rEV which is constant
    matA = numpy.concatenate((matA, [lEV]))
                
    phi = numpy.linalg.lstsq(numpy.transpose(matA), vecA)    
    phi = numpy.delete(phi[0], -1)
                    
    sensitivity = -numpy.outer(rEV, phi) + numpy.dot(phi,rEV) * numpy.outer(rEV, lEV)           
        
    return sensitivity

def mfpt_sensitivity(T, target, i):
    
    n = len(T)
    
    matA = T - numpy.diag(numpy.ones((n)))
    matA[target] *= 0
    matA[target, target] = 1.0
    
    tVec = -1. * numpy.ones(n);
    tVec[target] = 0;
    
    mfpt = numpy.linalg.solve(matA, tVec)
    aVec = numpy.zeros(n)
    aVec[i] = 1.0
    
    phiVec = numpy.linalg.solve(numpy.transpose(matA), aVec )
    
    # TODO: Check sign of sensitivity!
        
    sensitivity = -1.0 * numpy.outer(phiVec, mfpt)
    sensitivity[target] *= 0;
    
    return sensitivity