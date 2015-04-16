
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

import numpy as np
import sys

__author__ = 'Fabian Paul'
__all__ = ['acf']


def acf(trajs, stride=1, max_lag=None, subtract_mean=True, normalize=True, mean=None):
    '''Computes the (combined) autocorrelation function of multiple trajectories.

       Parameters
       ----------
       trajs : list of (*,N) ndarrays
         the observable trajectories, N is the number of observables
       stride : int (default = 1)
         only take every n'th frame from trajs
       max_lag : int (default = maximum trajectory length / stride)
         only compute acf up to this lag time
       subtract_mean : bool (default = True)
         subtract trajectory mean before computing acfs
       normalize : bool (default = True)
         divide acf be the variance such that acf[0,:]==1
       mean : (N) ndarray (optional)
         if subtract_mean is True, you can give the trajectory mean
         so this functions doesn't have to compute it again

       Returns
       -------
       acf : (max_lag,N) ndarray
           autocorrelation functions for all observables

       Note
       ----
       The computation uses FFT (with zero-padding) and is done im memory (RAM).
    '''
    if not isinstance(trajs, list):
        trajs = [trajs]

    mytrajs = [None] * len(trajs)
    for i in xrange(len(trajs)):
        if trajs[i].ndim == 1:
            mytrajs[i] = trajs[i].reshape((trajs[i].shape[0], 1))
        elif trajs[i].ndim == 2:
            mytrajs[i] = trajs[i]
        else:
            raise Exception(
                'Unexpected number of dimensions in trajectory number %d' % i)
    trajs = mytrajs

    assert stride > 0, 'stride must be > 0'
    assert max_lag is None or max_lag > 0, 'max_lag must be > 0'

    if subtract_mean and mean is None:
        # compute mean over all trajectories
        mean = trajs[0].sum(axis=0)
        n_samples = trajs[0].shape[0]
        for i, traj in enumerate(trajs[1:]):
            if traj.shape[1] != mean.shape[0]:
                raise Exception(('number of order parameters in trajectory number %d differs' +
                                 'from the number found in previous trajectories.') % (i + 1))
            mean += traj.sum(axis=0)
            n_samples += traj.shape[0]
        mean /= n_samples

    res = np.array([[]])
    # number of samples for every tau
    N = np.array([])

    for i, traj in enumerate(trajs):
        data = traj[::stride]
        if subtract_mean:
            data -= mean
        # calc acfs
        l = data.shape[0]
        fft = np.fft.fft(data, n=2 ** int(np.ceil(np.log2(l * 2 - 1))), axis=0)
        acftraj = np.fft.ifft(fft * np.conjugate(fft), axis=0).real
        # throw away acf data for long lag times (and negative lag times)
        if max_lag and max_lag < l:
            acftraj = acftraj[:max_lag, :]
        else:
            acftraj = acftraj[:l, :]
            if max_lag:
                sys.stderr.write(
                    'Warning: trajectory number %d is shorter than maximum lag.\n' % i)
        # find number of samples used for every lag
        Ntraj = np.linspace(l, l - acftraj.shape[0] + 1, acftraj.shape[0])
        # adapt shape of acf: resize temporal dimension, additionally set
        # number of order parameters of acf in the first step
        if res.shape[1] < acftraj.shape[1] and res.shape[1] > 0:
            raise Exception(('number of order parameters in trajectory number %d differs ' +
                             'from the number found in previous trajectories.') % i)
        if res.shape[1] < acftraj.shape[1] or res.shape[0] < acftraj.shape[0]:
            res.resize(acftraj.shape)
            N.resize(acftraj.shape[0])
        # update acf and number of samples
        res[0:acftraj.shape[0], :] += acftraj
        N[0:acftraj.shape[0]] += Ntraj

    # divide by number of samples
    res = np.transpose(np.transpose(res) / N)

    # normalize acfs
    if normalize:
        res /= res[0, :].copy()

    return res