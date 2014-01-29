'''
Created on Jan 8, 2014

@author: noe
'''

import numpy as np
import emma2.util.pystallone as stallone

def generate_traj(T, s, N, dt=1):
    """
    Generates a realization of the Markov chain with transition matrix T
    
    Parameters
    ----------
    T: ndarray
        transition matrix
    s: int
        starting state
    N: int
        trajectory length
    dt: int
        number of steps before saving a state. 
    
    returns: A discrete trajectory of length will be N/dt
    """
    Tstall = stallone.ndarray_to_stallone_array(T)
    mc = stallone.API.msmNew.markovChain(Tstall)
    mc.setStartingState(s)
    traj_full = mc.randomTrajectory(N).getArray()
    traj_sliced = (np.array(traj_full))[0:len(traj_full):dt]
    return traj_sliced

# TODO: Implement in python or call to stallone (FRANK)
def trajectory_generator(T, start=None, stop=None):
    """
    Generates a markov chain object whose purpose is to generate trajectories
    
    The returned object will is able to efficiently generate trajectories
    from the Markov chain with transition matrix T. Its main function is the
    following:
        trajectory(length=1000, start=None, skip=1, out=None)
    which generates an integer sequence of length length/skip from the 
    specified starting state (when not given will be drawn from the
    starting_distribution). The trajectory is of the specified length, but
    when skip > 1, only every skip frames will be returned. When out is 
    specified, the output trajectory will be written there
    
    Parameters
    ----------
    start : int or array
        distribution from which the initial state of generated trajectories
        will be drawn. A single int specifies a deterministic starting state.
        When set to None (default), the uniform distribution will be used. This
        parameter is overridden by specifying a specific start state in the
        call to trajectory(...)
    stop : int or sequence of ints
        trajectory will stop before reaching length when it hits one of the
        specified states. If it is desired that trajectories should always
        run until the stop state(s) are hit, then set the trajectory length
        to infinite. Note that this might produce a memory error though, when
        trajectories become too long to be stored.
    """

# TODO: Implement in python or call to stallone (FRANK)
def trajectory(T, length=1000, start=None, stop=None, skip=1, out=None):
    """
    Generates a Markov chain trajectory using transition matrix T
    
    The starting state is either deterministic (fixed by setting the start-
    value to an integer < n, where n is the row/column-size of T, 
    e.g. 'start=0') or drawn from a user-specified distribution 
    of length n (e.g. 'start=numpy.array([0.5, 0.4, 0.1])'.
    The trajectory will be run until it has reached the specified
    length, or until it has hit a state from the set of stop states
    specified.
    
    Parameters
    ----------
    T : array
        sparse or dense quadratic array (size n x n)
    start: int or array
        when set to None, the starting state will be chosen uniformly random
        when set to an int, the trajectory will start with this state.
        when set to a n-sized array, the starting state will be randomly
        generated from this distribution.
    stop : int or sequence of ints
        trajectory will stop before reaching length when it hits one of the
        specified states. If it is desired that trajectories should always
        run until the stop state(s) are hit, then set the trajectory length
        to infinite. Note that this might produce a memory error though, when
        trajectories become too long to be stored.
    
    Returns
    -------
    traj : array
        an array of size length/skip. When stop is set, then the returned
        trajectory is shorter than length/skip when one of the stopping states
        has been hit
    
    Raises
    -------
    ValueError
        If T is nonquadratic, if starting distribution has the wrong length,
        if start or stop states have been specified outside [0,n-1].
    """
    