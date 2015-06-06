
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

@author: noe
'''
__docformat__ = "restructuredtext en"

__all__ = ['ImpliedTimescales']

import numpy as np
import warnings

from pyemma.util.statistics import confidence_interval
from pyemma.util.discrete_trajectories import number_of_states
from pyemma.util import types as _types
from pyemma._base.estimator import Estimator, get_estimator, param_grid, estimate_param_scan


# ====================================================================
# Helper functions
# ====================================================================

def _generate_lags(maxlag, multiplier):
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


# TODO: connectivity flag is currently not used. Introduce different connectivity modes (lag, minimal, set)
# TODO: if not connected, might add infinity timescales.
# TODO: Timescales should be assigned by similar eigenvectors rather than by order
# TODO: when requesting too long lagtimes, throw a warning and exclude lagtime from calculation, but compute the rest
class ImpliedTimescales(Estimator):
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
    failfast = False : boolean
        if True, will raise an error as soon as not all requested timescales can be computed at all requested
        lagtimes. If False, will continue with a warning and compute the timescales/lagtimes that are possible.

    """
    def __init__(self, data, estimator, lags=None, nits=10, failfast=False):
        # initialize
        self.data = data
        self.estimator = get_estimator(estimator)

        # estimated its. 2D-array with indexing: lagtime, its
        self._its = None
        # sampled its's. 3D-array with indexing: lagtime, its, sample
        self._its_samples = None

        # trajectory lengths
        try:
            self.data = _types.ensure_dtraj_list(data)
            # maximum number of timescales
            nstates = number_of_states(self.data)
            self.nits = min(nits, nstates - 1)
            #
            self.lengths = np.zeros(nits)
            for i in range(len(self.data)):
                self.lengths[i] = len(self.data[i])
            self.maxlength = np.max(self.lengths)
        except:
            raise NotImplementedError('Currently only discrete trajectories are implemented')

        # lag time
        if lags is None:
            maxlag = 0.5 * np.sum(self.lengths) / float(len(self.lengths))
            self.lags = _generate_lags(maxlag, 1.5)
        else:
            self.lags = np.array(lags)
            # check if some lag times are forbidden.
            if np.max(self.lags) >= self.maxlength:
                Ifit = np.where(self.lags < self.maxlength)[0]
                Inofit = np.where(self.lags >= self.maxlength)[0]
                warnings.warn(
                    'Some lag times exceed the longest trajectories. Will ignore lag times: ' + str(self.lags[Inofit]))
                self.lags = self.lags[Ifit]

    def _log_no_ts(self, tau):
        warnings.warn('Could not compute a single timescale at tau = ' + str(tau) +
                      '. Probably a connectivity problem. Try using smaller lagtimes')

    def _estimate(self, data):
        r"""Estimates ITS at set of lagtimes
        
        """
        # construct all parameter sets for the estimator
        param_sets = param_grid({'lag': self.lags})

        # run estimation on all lag times
        estimates = estimate_param_scan(self.estimator, data, param_sets,
                                        evaluate=['timescales', 'timescales_samples'], failfast=False)

        # store timescales
        timescales = [est[0] for est in estimates]
        if all(ts is None for ts in timescales):
            raise RuntimeError('Could not compute any timescales. Make sure that your estimator object does provide'
                               'the timescales method or property')
        nits = min(max([len(ts) for ts in timescales]), self.nits)
        self._its = np.zeros((len(self.lags), nits))
        for i, ts in enumerate(timescales):
            if ts is None:
                self._log_no_ts(self.lags[i])
            else:
                ts = ts[:nits]
                self._its[i,:len(ts)] = ts  # copy into array. Leave 0 if there is no timescales

        # store timescales samples (if available)
        timescales_samples = [est[1] for est in estimates]
        if all(ts is None for ts in timescales_samples):
            self._its_samples = None
        else:
            nits = min(max([np.shape(ts)[1] for ts in timescales_samples]), self.nits)
            nsamples = np.shape(timescales_samples[0])[0]
            self._its_samples = np.zeros((len(self.lags), nits, nsamples))
            for i, ts in enumerate(timescales_samples):
                if ts is None:
                    self._log_no_ts(self.lags[i])
                else:
                    ts = ts[:nits]
                    self._its_samples[i,:ts.shape[1],:] = ts  # copy into array. Leave 0 if there is no timescales

    @property
    def lagtimes(self):
        r"""Return the list of lag times for which timescales were computed.

        """
        return self.lags

    def get_lagtimes(self):
        r"""Return the list of lag times for which timescales were computed.
        
        """
        warnings.warn('get_lagtimes() is deprecated. Use lagtimes', DeprecationWarning)
        return self.lags

    @property
    def number_of_timescales(self):
        r"""Return the number of timescales.
        
        """
        return self.nits

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
        return self.lags
        # return self._lags_sample

    @property
    def sample_number_of_timescales(self):
        r"""Return the number of timescales for which sample data is available

        """
        return self.nits
        # return self._nits_sample

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
            raise RuntimeError('Cannot compute sample mean, because no samples were generated ' +
                               ' try calling bootstrap() before')
        # OK, go:
        if process is None:
            L = np.zeros((len(self.lags), self.nits))
            R = np.zeros((len(self.lags), self.nits))
            for i in range(len(self.lags)):
                for j in range(self.nits):
                    conf = confidence_interval(self._its_samples[i, j], alpha)
                    L[i, j] = conf[1]
                    R[i, j] = conf[2]
            return L, R
        else:
            L = np.zeros(len(self.lags))
            R = np.zeros(len(self.lags))
            for i in range(len(self.lags)):
                conf = confidence_interval(self._its_samples[i, process], alpha)
                L[i] = conf[1]
                R[i] = conf[2]
            return L, R