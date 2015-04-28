
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on Jan 8, 2014

@author: noe
'''

import math
import numpy as np
import scipy.stats
import pyemma.util.types as types

__all__ = ['transition_matrix_metropolis_1d',
           'generate_traj',
           'generate_trajs']


class MarkovChainSampler:
    """
    Class for generation of trajectories from a transition matrix P.
    If many trajectories will be sampled from P, using this class is much more efficient than individual calls to
    generate_traj because that avoid costly multiple construction of random variable objects.

    """

    def __init__(self, P, dt=1):
        """
        Constructs a sampling object with transition matrix P. The results will be produced every dt'th time step

        Parameters
        ----------
        P : (n, n) ndarray
            transition matrix
        dt : int
            trajectory will be saved every dt time steps.
            Internally, the dt'th power of P is taken to ensure a more efficient simulation.

        """
        # process input
        if dt > 1:
            # take a power of P if requested
            self.P = np.linalg.matrix_power(P, dt)
        else:
            # create local copy and transform to ndarray if in a different format
            self.P = np.array(P)
        self.n = self.P.shape[0]

        # initialize mu
        self.mudist = None

        # generate discrete random value generators for each line
        self.rgs = np.ndarray((self.n), dtype=object)
        for i in range(self.n):
            nz = np.nonzero(self.P[i])
            self.rgs[i] = scipy.stats.rv_discrete(values=(nz, self.P[i, nz]))

    def trajectory(self, N, start=None, stop=None):
        """
        Generates a trajectory realization of length N, starting from state s

        Parameters
        ----------
        N : int
            trajectory length
        start : int, optional, default = None
            starting state. If not given, will sample from the stationary distribution of P
        stop : int or int-array-like, optional, default = None
            stopping set. If given, the trajectory will be stopped before N steps
            once a state of the stop set is reached

        """
        # check input
        stop = types.ensure_int_array_or_None(stop, require_order=False)

        if start is None:
            if self.mudist is None:
                # compute mu, the stationary distribution of P
                import pyemma.msm.analysis as msmana

                mu = msmana.stationary_distribution(self.P)
                self.mudist = scipy.stats.rv_discrete(values=(range(self.n), mu ))
            # sample starting point from mu
            start = self.mudist.rvs()

        # evaluate stopping set
        stopat = np.ndarray((self.n), dtype=bool)
        stopat[:] = False
        if (stop is not None):
            for s in np.array(stop):
                stopat[s] = True

        # result
        traj = np.zeros(N, dtype=int)
        traj[0] = start
        # already at stopping state?
        if stopat[traj[0]]:
            return traj[:1]
        # else run until end or stopping state
        for t in range(1, N):
            traj[t] = self.rgs[traj[t - 1]].rvs()
            if stopat[traj[t]]:
                return traj[:t+1]
        # return
        return traj

    def trajectories(self, M, N, start=None, stop=None):
        """
        Generates M trajectories, each of length N, starting from state s

        Parameters
        ----------
        M : int
            number of trajectories
        N : int
            trajectory length
        start : int, optional, default = None
            starting state. If not given, will sample from the stationary distribution of P
        stop : int or int-array-like, optional, default = None
            stopping set. If given, the trajectory will be stopped before N steps
            once a state of the stop set is reached

        """
        trajs = [self.trajectory(N, start=start, stop=stop) for i in range(M)]
        return trajs

def generate_traj(P, N, start=None, stop=None, dt=1):
    """
    Generates a realization of the Markov chain with transition matrix P.

    Parameters
    ----------
    P : (n, n) ndarray
        transition matrix
    N : int
        trajectory length
    start : int, optional, default = None
        starting state. If not given, will sample from the stationary distribution of P
    stop : int or int-array-like, optional, default = None
        stopping set. If given, the trajectory will be stopped before N steps
        once a state of the stop set is reached
    dt : int
        trajectory will be saved every dt time steps.
        Internally, the dt'th power of P is taken to ensure a more efficient simulation.

    Returns
    -------
    traj_sliced : (N/dt, ) ndarray
        A discrete trajectory with length N/dt

    """
    sampler = MarkovChainSampler(P, dt=dt)
    return sampler.trajectory(N, start=start, stop=stop)


def generate_trajs(P, M, N, start=None, stop=None, dt=1):
    """
    Generates multiple realizations of the Markov chain with transition matrix P.

    Parameters
    ----------
    P : (n, n) ndarray
        transition matrix
    M : int
        number of trajectories
    N : int
        trajectory length
    start : int, optional, default = None
        starting state. If not given, will sample from the stationary distribution of P
    stop : int or int-array-like, optional, default = None
        stopping set. If given, the trajectory will be stopped before N steps
        once a state of the stop set is reached
    dt : int
        trajectory will be saved every dt time steps.
        Internally, the dt'th power of P is taken to ensure a more efficient simulation.

    Returns
    -------
    traj_sliced : (N/dt, ) ndarray
        A discrete trajectory with length N/dt

    """
    sampler = MarkovChainSampler(P, dt=dt)
    return sampler.trajectories(M, N, start=start, stop=stop)


def transition_matrix_metropolis_1d(E, d=1.0):
    r"""Transition matrix describing the Metropolis chain jumping
    between neighbors in a discrete 1D energy landscape.
    
    Parameters
    ----------
    E : (M,) ndarray
        Energies in units of kT
    d : float (optional)
        Diffusivity of the chain, d in (0, 1]
        
    Returns
    -------
    P : (M, M) ndarray
        Transition matrix of the Markov chain

    Notes
    -----
    Transition probabilities are computed as p_i,i+1 = 0.5 * d *
    min(1.0, exp(-(E_i+1 - E_i))).
        
    """
    # check input
    if (d <= 0 or d > 1):
        raise ValueError('Diffusivity must be in (0,1]. Trying to set the invalid value', str(d))
    # init
    n = len(E)
    P = np.zeros((n, n))
    # set offdiagonals
    P[0, 1] = 0.5 * d * min(1.0, math.exp(-(E[1] - E[0])))
    for i in range(1, n - 1):
        P[i, i - 1] = 0.5 * d * min(1.0, math.exp(-(E[i - 1] - E[i])))
        P[i, i + 1] = 0.5 * d * min(1.0, math.exp(-(E[i + 1] - E[i])))
    P[n - 1, n - 2] = 0.5 * d * min(1.0, math.exp(-(E[n - 2] - E[n - 1])))
    # normalize
    P += np.diag(1.0 - np.sum(P, axis=1))
    # done
    return P