'''
Created on Jul 23, 2014

@author: noe
'''

import numpy as np
import math
from scipy.stats import rv_discrete
import random
from emma2.msm.estimation import count_matrix

def determine_lengths(dtrajs):
    """
    Determines the lengths of all trajectories
    """
    if (isinstance(dtrajs[0], (int,long))):
        return [len(dtrajs)]
    lengths = []
    for dtraj in dtrajs:
        lengths.append(len(dtraj))
    return lengths

def bootstrap_trajectories(trajs, correlation_length):
    """
    Generates a randomly resampled count matrix given the input coordinates.

    See API function for full documentation.
    """
    # if we have just one trajectory, put it into a one-element list:
    if (isinstance(trajs[0], (int, long))):
        trajs = [trajs]
    ntraj = len(trajs)
    
    # determine actual correlation length
    lengths = determine_lengths(trajs)
    Lmax = np.max(lengths)
    Ltot = np.sum(lengths)
    if (correlation_length < 1): # use full trajectories
        correlation_length = Lmax
    if (correlation_length >= Ltot):
        raise Warning('correlation length is greater than the total amount of data. This will not sample '+
                      'the count matrix, but reproducibly give the same count matrix as direct counting.')
        
    # assigning trajectory sampling weights
    w_trajs = np.zeros((len(trajs)))
    for i in range(ntraj):
        if len(trajs[i]) < correlation_length:
            # this trajectory can only be used in full, but counts with a smaller weight
            w_trajs[i] = 1.0 * len(trajs[i]) / correlation_length 
        else:
            # weight equal to the number of available starting points.
            w_trajs[i] = len(trajs[i]) - correlation_length
    w_trajs /= np.sum(w_trajs) # normalize to sum 1.0
    distrib_trajs = rv_discrete(values=(range(ntraj), w_trajs))
    
    # generate subtrajectories
    nsubs = int(max(1, math.floor(Ltot / correlation_length)))
    subs = []    
    for i in range(nsubs):
        # pick a random trajectory
        itraj = distrib_trajs.rvs()
        # pick a starting frame
        t0 = random.randint(0, max(1,len(trajs[itraj])-correlation_length))
        t1 = min(len(trajs[itraj]), t0+correlation_length)
        # add new subtrajectory
        subs.append(trajs[itraj][t0:t1])

    # and return
    return subs


def bootstrap_counts(dtrajs, correlation_length, lagtime):
    """
    Generates a randomly resampled count matrix given the input coordinates.
    
    See API function for full documentation.
    """
    # correlation length too short? use at least lagtime
    if (correlation_length < lagtime):
        correlation_length = lagtime    
    # can we do the estimate?
    lengths = determine_lengths(dtrajs)
    Lmax = np.max(lengths)
    if (lagtime > Lmax):
        raise ValueError('Cannot estimate count matrix: lag time '
                         +str(lagtime)+' is longer than the longest trajectory length'+str(Lmax))

    # generate bootstrapped subtrajectories of correlation length
    subs = bootstrap_trajectories(dtrajs, correlation_length)
    # generate count matrix
    Z = count_matrix(subs, lagtime)    
    # and return
    return Z