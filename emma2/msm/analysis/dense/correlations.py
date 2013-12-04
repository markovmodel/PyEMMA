'''
Created on 29.11.2013

@author: marscher
'''

def crosscorrelation(P, obs1, obs2):
    from decomposition import stationary_distribution as sdist
    sum_ = 0.0
    pi = sdist(P)
    for i in range(P.shape[0]):
        for j in range(P.shape[0]):
            # TODO: maybe this could be written in a more convenient way
            sum_ += obs1[i] * pi[i] * P[i][j] * obs2[j]
    return sum_