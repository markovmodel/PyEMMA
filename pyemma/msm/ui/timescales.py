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
Created on Jul 26, 2014

@author: noe, marscher
'''

import numpy as np
import warnings
import multiprocessing

from pyemma.msm.estimation import cmatrix, connected_cmatrix, tmatrix, bootstrap_counts
from pyemma.msm.analysis import timescales
from pyemma.util.statistics import confidence_interval
from pyemma.util.types import ensure_dtraj_list as _ensure_dtraj_list
from pyemma.util.discrete_trajectories import number_of_states
from pyemma.util.progressbar import ProgressBar
from pyemma.util.progressbar.gui import show_progressbar

__docformat__ = "restructuredtext en"
__all__ = ['ImpliedTimescales']


# helper functions
def _estimate_ts_tau(C, tau, reversible, nits):
    r"""Estimate timescales from the given count matrix.

    """
    # connected set
    C = connected_cmatrix(C).toarray()
    if (len(C) > 1):
        # estimate transition matrix
        T = tmatrix(C, reversible=reversible)
        # timescales
        ts = timescales(T, tau, k=min(nits, len(T)) + 1)[1:]
        return ts
    else:
        return None  # no timescales available


def _estimate(tau, dtrajs, reversible, nits):
    C = cmatrix(dtrajs, tau)
    # estimate timescales
    ts = _estimate_ts_tau(C, tau, reversible, nits)
    return tau, ts


def _estimate_bootstrap(tau, dtrajs, reversible, nits, parallel=True):
    if parallel:
        # if we are in a parallel process set seed to get distinct order of random numbers
        # otherwise, we would get the same samples
        np.random.seed(None)
    # sample count matrix
    C = bootstrap_counts(dtrajs, tau)
    # estimate timescales
    ts = _estimate_ts_tau(C, tau, reversible, nits)
    return tau, ts


# TODO: connectivity flag is currently not used. Introduce different connectivity modes (lag, minimal, set)
# TODO: if not connected, might add infinity timescales.
# TODO: Timescales should be assigned by similar eigenvectors rather than by order
# TODO: when requesting too long lagtimes, throw a warning and exclude
# lagtime from calculation, but compute the rest
class ImpliedTimescales(object):
    r"""Implied timescales for a series of lag times.

    Parameters
    ----------
    dtrajs : array-like or list of array-likes
        discrete trajectories
    lags = None : array-like with integers
        integer lag times at which the implied timescales will be calculated
    k = 10 : int
        number of implied timescales to be computed. Will compute less if the number of
        states are smaller
    connected = True : boolean
        compute the connected set before transition matrix estimation at each lag
        separately
    reversible = True : boolean
        estimate the transition matrix reversibly (True) or nonreversibly (False)
    failfast = False : boolean
        if True, will raise an error as soon as not all requested timescales can be computed at all requested
        lagtimes. If False, will continue with a warning and compute the timescales/lagtimes that are possible.
    n_procs : int (default=1 CPU)
        parallelize estimation of timescales over given count of CPU cores (subprocessess).
        Pass None to utilize all cores.

    """

    def __init__(self, dtrajs, lags=None, nits=10, connected=True,
                 reversible=True, failfast=False, n_procs=1):
        # initialize
        self._dtrajs = _ensure_dtraj_list(dtrajs)
        self._connected = connected
        self._reversible = reversible

        # maximum number of timescales
        nstates = number_of_states(self._dtrajs)
        self._nits = min(nits, nstates - 1)

        # trajectory lengths
        self.lengths = np.zeros(len(self._dtrajs))
        for i in range(len(self._dtrajs)):
            self.lengths[i] = len(self._dtrajs[i])
        self.maxlength = np.max(self.lengths)

        # lag time
        if lags is None:
            maxlag = 0.5 * np.sum(self.lengths) / float(len(self.lengths))
            self._lags = self._generate_lags(maxlag, 1.5)
        else:
            self._lags = np.array(lags)
            # check if some lag times are forbidden.
            if np.max(self._lags) >= self.maxlength:
                Ifit = np.where(self._lags < self.maxlength)[0]
                Inofit = np.where(self._lags >= self.maxlength)[0]
                warnings.warn(
                    'Some lag times exceed the longest trajectories. '
                    'Will ignore lag times: ' + str(self._lags[Inofit]))
                self._lags = self._lags[Ifit]

        if n_procs is None:
            self._n_procs = multiprocessing.cpu_count()
        else:
            self._n_procs = n_procs
        if n_procs > 1:
            self._pool = multiprocessing.Pool(self._n_procs)

        # estimate
        self._estimate()

    def _generate_lags(self, maxlag, multiplier):
        r"""Generate a set of lag times starting from 1 to maxlag,
        using the given multiplier between successive lags

        """
        # determine lag times
        lags = []
        # build default lag list
        lags.append(1)
        lag = 1.0
        while (lag <= maxlag):
            lag = round(lag * multiplier)
            lags.append(int(lag))
        return lags

    def _estimate(self):
        r"""Estimates ITS at set of lagtimes

        """
        # initialize
        self._its = np.zeros((len(self._lags), self._nits))
        maxnits = self._nits
        maxnlags = len(self._lags)

        pg = ProgressBar(len(self._its), description="calc timescales")

        no_single_ts_msg = 'Could not compute a single timescale at tau = %i' \
                           '. Probably a connectivity problem. Try using smaller lagtimes'

        no_ts_at_tau_msg = 'Could only compute %i timescales at tau = %i' \
                           ' instead of the requested %s. Probably a ' \
                           'connectivity problem. Request less timescales or smaller lagtimes'

        if self._n_procs > 1:
            results = []
            jobs = []
            def _callback(X):
                results.append(X)
                pg.numerator += 1
                show_progressbar(pg)

        for i in xrange(len(self._lags)):
            # get lag time to be used
            tau = self._lags[i]
            args = (tau, self._dtrajs, self._reversible, self._nits)
            if self._n_procs > 1:
                j = self._pool.apply_async(_estimate, args=args, callback=_callback)
                jobs.append(j)
            else:
                _, ts = _estimate(*args)

                if ts is None:
                    maxnlags = i
                    warnings.warn(no_single_ts_msg % tau)
                    break
                elif len(ts) < self._nits:
                    maxnits = min(maxnits, len(ts))
                    warnings.warn(no_ts_at_tau_msg % (len(ts), tau, self._nits))
                self._its[i, :] = ts
                pg.numerator += 1
                show_progressbar(pg)

        # parallel results assignment
        if self._n_procs > 1:
            # wait for all jobs to finish their calculation
            for j in jobs:
                j.wait()
            # sort by lag time
            results.sort(key=lambda x: x[0])
            for i, res in enumerate(results):
                tau, ts = res
                if ts is None:
                    maxnlags = i
                    warnings.warn(no_single_ts_msg % tau)
                    break
                elif len(ts) < self._nits:
                    maxnits = min(maxnits, len(ts))
                    warnings.warn(no_ts_at_tau_msg % (len(ts), tau, self._nits))
                self._its[i, :] = ts

        # any infinities?
        if np.any(np.isinf(self._its)):
            warnings.warn('Timescales contain infinities, indicating that the'
                          ' data is disconnected at some lag time')

        # clean up
        self._nits = maxnits
        self._lags = self._lags[:maxnlags]
        self._its = self._its[:maxnlags][:, :maxnits]

    def bootstrap(self, nsample=10):
        r"""Samples ITS using bootstrapping

        Parameters
        ----------
        nsample : int
            bootstrap using n samples

        """
        # initialize
        self._its_samples = np.zeros((len(self._lags), self._nits, nsample))
        self._nits_sample = self._nits
        maxnits = self._nits_sample
        maxnlags = len(self._lags)

        denom = self._its_samples.shape[0] * nsample
        pg = ProgressBar(denom, description="bootstrapping timescales")

        if self._n_procs > 1:
            jobs = []
            results = []

            def _callback(X):
                results.append(X)
                pg.numerator += 1
                show_progressbar(pg)

        for i in xrange(len(self._lags)):
            tau = self._lags[i]
            all_ts = True
            any_ts = True

            for k in xrange(nsample):
                args = (tau, self._dtrajs, self._reversible, self._nits)
                if self._n_procs > 1:
                    j = self._pool.apply_async(_estimate_bootstrap,
                                               args=args, callback=_callback)
                    jobs.append(j)
                else:
                    _, ts = _estimate_bootstrap(*args, parallel=False)
                    # only use ts if we get all requested timescales
                    if ts is not None:
                        if len(ts) == self._nits:
                            self._its_samples[i, :, k] = ts
                        else:
                            all_ts = False
                            maxnits = min(maxnits, len(ts))
                            self._its_samples[i, :maxnits, k] = ts[:maxnits]
                    else:
                        any_ts = False
                        maxnlags = i
                    pg.numerator += 1
                    show_progressbar(pg)

        if self._n_procs > 1:
            # wait for all jobs to finish
            for j in jobs:
                j.wait()

            # post-processing
            # sort inplace according to tau
            results.sort(key=lambda x: x[0])

            samples_by_lag = {tau: [] for tau in self._lags}
            for tau, ts in results:
                samples_by_lag[tau].append(ts)

            for ii, tau in enumerate(self._lags):
                # iterate over samples
                for jj, ts in enumerate(samples_by_lag[tau]):
                    if ts is not None:
                        if len(ts) == self._nits:
                            self._its_samples[ii, :, jj] = ts
                        else:
                            all_ts = False
                            maxnits = min(maxnits, len(ts))
                            self._its_samples[ii, :maxnits, jj] = ts[:maxnits]
                    else:
                        any_ts = False
                        maxnlags = i

        if not all_ts:
            warnings.warn('Could not compute all requested timescales at tau = ' + str(tau) +
                          '. Bootstrap is incomplete and might be non-representative.' +
                          ' Request less timescales or smaller lagtimes')
        if not any_ts:
            warnings.warn('Could not compute a single timescale at tau = ' + str(tau) +
                          '. Probably a connectivity problem. Try using smaller lagtimes')
        # clean up
        self._nits_sample = maxnits
        self._lags_sample = self._lags[:maxnlags]
        self._its_samples = self._its_samples[:maxnlags, :, :][:, :maxnits, :]

    @property
    def lagtimes(self):
        r"""Return the list of lag times for which timescales were computed.

        """
        return self._lags

    def get_lagtimes(self):
        r"""Return the list of lag times for which timescales were computed.

        """
        warnings.warn(
            'get_lagtimes() is deprecated. Use lagtimes', DeprecationWarning)
        return self._lags

    @property
    def number_of_timescales(self):
        r"""Return the number of timescales.

        """
        return self._nits

    @property
    def timescales(self):
        r"""Returns the implied timescale estimates

        Returns
        --------
        timescales : ndarray((l x k), dtype=float)
            timescales for all processes and lag times.
            l is the number of lag times and k is the number of computed timescales.

        """
        return self.get_timescales()

    def get_timescales(self, process=None):
        r"""Returns the implied timescale estimates

        Parameters
        ----------
        process : int or None (default)
            index in [0:n-1] referring to the process whose timescale will be returned.
            By default, process = None and all computed process timescales will be returned.

        Returns
        --------
        if process is None, will return a (l x k) array, where l is the number of lag times 
        and k is the number of computed timescales.
        if process is an integer, will return a (l) array with the selected process time scale
        for every lag time

        """
        if process is None:
            return self._its
        else:
            return self._its[:, process]

    @property
    def samples_available(self):
        r"""Returns True if samples are available and thus sample
        means, standard errors and confidence intervals can be
        obtained

        """
        return self._its_samples is not None

    @property
    def sample_lagtimes(self):
        r"""Return the list of lag times for which sample data is available

        """
        return self._lags_sample

    @property
    def sample_number_of_timescales(self):
        r"""Return the number of timescales for which sample data is available

        """
        return self._nits_sample

    @property
    def sample_mean(self):
        r"""Returns the sample means of implied timescales. Need to
        generate the samples first, e.g. by calling bootstrap

        Returns
        -------
        timescales : ndarray((l x k), dtype=float)
            mean timescales for all processes and lag times.
            l is the number of lag times and k is the number of computed timescales.

        """
        return self.get_sample_mean()

    def get_sample_mean(self, process=None):
        r"""Returns the sample means of implied timescales. Need to
        generate the samples first, e.g. by calling bootstrap

        Parameters
        ----------
        process : int or None (default)
            index in [0:n-1] referring to the process whose timescale will be returned.
            By default, process = None and all computed process timescales will be returned.

        Returns
        -------
        if process is None, will return a (l x k) array, where l is the number of lag times 
        and k is the number of computed timescales.
        if process is an integer, will return a (l) array with the selected process time scale
        for every lag time

        """
        if self._its_samples is None:
            raise RuntimeError('Cannot compute sample mean, because no samples were generated ' +
                               ' try calling bootstrap() before')
        # OK, go:
        if process is None:
            return np.mean(self._its_samples, axis=2)
        else:
            return np.mean(self._its_samples[:, process, :], axis=1)

    @property
    def sample_std(self):
        r"""Returns the standard error of implied timescales. Need to
        generate the samples first, e.g. by calling bootstrap


        Returns
        -------
        Returns
        -------
        timescales : ndarray((l x k), dtype=float)
            standard deviations of timescales for all processes and lag times.
            l is the number of lag times and k is the number of computed timescales.

        """
        return self.get_sample_std()

    def get_sample_std(self, process=None):
        r"""Returns the standard error of implied timescales. Need to
        generate the samples first, e.g. by calling bootstrap

        Parameters
        -----------
        process : int or None (default)
            index in [0:n-1] referring to the process whose timescale will be returned.
            By default, process = None and all computed process timescales will be returned.

        Returns
        -------
        if process is None, will return a (l x k) array, where l is the number of lag times 
        and k is the number of computed timescales.
        if process is an integer, will return a (l) array with the selected process time scale
        for every lag time

        """
        if self._its_samples is None:
            raise RuntimeError('Cannot compute sample mean, because no samples were generated ' +
                               ' try calling bootstrap() before')
        # OK, go:
        if process is None:
            return np.std(self._its_samples, axis=2)
        else:
            return np.std(self._its_samples[:, process, :], axis=1)

    def get_sample_conf(self, alpha=0.6827, process=None):
        r"""Returns the confidence interval that contains alpha % of the sample data

        Use:
        alpha = 0.6827 for 1-sigma confidence interval
        alpha = 0.9545 for 2-sigma confidence interval
        alpha = 0.9973 for 3-sigma confidence interval
        etc.

        Returns
        -------
        (L,R) : (float[],float[]) or (float[][],float[][])
            lower and upper timescales bounding the confidence interval
        if process is None, will return two (l x k) arrays, where l is the number of lag times 
        and k is the number of computed timescales.
        if process is an integer, will return two (l)-arrays with the
        selected process time scale for every lag time

        """
        if self._its_samples is None:
            raise RuntimeError('Cannot compute sample mean, because no samples'
                               ' were generated try calling bootstrap() before')
        # OK, go:
        if process is None:
            L = np.zeros((len(self._lags), self._nits))
            R = np.zeros((len(self._lags), self._nits))
            for i in xrange(len(self._lags)):
                for j in xrange(self._nits):
                    conf = confidence_interval(self._its_samples[i, j], alpha)
                    L[i, j] = conf[1]
                    R[i, j] = conf[2]
            return (L, R)
        else:
            L = np.zeros(len(self._lags))
            R = np.zeros(len(self._lags))
            for i in xrange(len(self._lags)):
                conf = confidence_interval(
                    self._its_samples[i, process], alpha)
                L[i] = conf[1]
                R[i] = conf[2]
            return (L, R)
