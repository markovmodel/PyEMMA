'''
Created on 29.11.2013

@author: marscher
'''

def crosscorrelation(P, obs1, obs2):
    from decomposition import stationary_distribution as sdist
    pi = sdist(P)
    # non zero indices of P
    inds = P.indices
    sum_ = 0.0
    for i in range(inds):
        for j in range(inds):
            # TODO: maybe this could be written in a more convenient way
            sum_ += obs1[i] * pi[i] * P[i][j] * obs2[j]
            
    return sum_