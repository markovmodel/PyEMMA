
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
from pyemma.util import types as _types
from pyemma._base.estimator import Estimator, get_estimator, param_grid, estimate_param_scan
from pyemma._base.model import SampledModel


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
    while lag <= maxlag:
        lag = round(lag * multiplier)
        if lag <= maxlag:
            lags.append(int(lag))
    return np.array(lags)


# TODO: build a generic implied timescales estimate in _base, and make this a subclass (for dtrajs)
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

    nits = 10 : int
        maximum number of implied timescales to be computed and stored. If less
        timescales are available, nits will be set to a smaller value during
        estimation

    failfast = False : boolean
        if True, will raise an error as soon as not all requested timescales can be computed at all requested
        lagtimes. If False, will continue with a warning and compute the timescales/lagtimes that are possible.

    """
    def __init__(self, estimator, lags=None, nits=10, failfast=False):
        # initialize
        self.estimator = get_estimator(estimator)
        self.nits = nits
        self.failfast = failfast

        # set lag times
        if _types.is_int(lags):  # got a single integer. We create a list
            self._lags = _generate_lags(lags, 1.5)
        else:  # got a list of ints or None - otherwise raise exception.
            self._lags = _types.ensure_int_vector_or_None(lags, require_order=True)

        print 'Generated lags: ', self._lags

        # estimated its. 2D-array with indexing: lagtime, its
        self._its = None
        # sampled its's. 3D-array with indexing: lagtime, its, sample
        self._its_samples = None


    def _estimate(self, data):
        r"""Estimates ITS at set of lagtimes
        
        """
        ### PREPARE AND CHECK DATA
        # TODO: Currenlty only discrete trajectories are implemented. For a general class this needs to be changed.
        data = _types.ensure_dtraj_list(data)

        # check trajectory lengths
        self._trajlengths = np.array([len(traj) for traj in data])
        maxlength = np.max(self._trajlengths)

        # set lag times by data if not yet set
        if self._lags is None:
            maxlag = 0.5 * np.sum(self._trajlengths) / float(len(self._trajlengths))
            self._lags = _generate_lags(maxlag, 1.5)

        # check if some lag times are forbidden.
        if np.max(self._lags) >= maxlength:
            Ifit = np.where(self._lags < maxlength)[0]
            Inofit = np.where(self._lags >= maxlength)[0]
            self.logger.warn('Ignoring lag times that exceed the longest trajectory: ' + str(self._lags[Inofit]))
            self._lags = self._lags[Ifit]

        ### RUN ESTIMATION

        # construct all parameter sets for the estimator
        param_sets = param_grid({'lag': self._lags})
        param_sets = [p for p in param_sets]

        # run estimation on all lag times
        self._models, self._estimators = estimate_param_scan(self.estimator, data, param_sets, return_estimators=True)

        ### PROCESS RESULTS

        # timescales
        timescales = [m.timescales() for m in self._models]

        # how many finity timescales do we really have?
        maxnts = max([len(ts[np.isfinite(ts)]) for ts in timescales])
        if maxnts < self.nits:
            self.nits = maxnts
            self.logger.warn('Changed user setting nits to the number of available timescales nits=' + str(self.nits))

        # sort timescales into matrix
        computed_all = True  # flag if we have found any problems
        self._its = np.empty((len(self._lags), self.nits))
        self._its[:] = np.NAN  # initialize with NaN in order to point out timescales that were not computed
        self._successful_lag_indexes = []
        for i, ts in enumerate(timescales):
            if ts is not None:
                if np.any(np.isfinite(ts)):  # if there are any finite timescales available, add them
                    self._its[i, :len(ts)] = ts[:self.nits]  # copy into array. Leave NaN if there is no timescale
                    self._successful_lag_indexes.append(i)

        if len(self._successful_lag_indexes) < len(self._lags):
            computed_all = False
        if np.any(np.isnan(self._its)):
            computed_all = False

        # timescales samples if available
        if issubclass(self._models[0].__class__, SampledModel):
            # samples
            timescales_samples = [m.sample_f('timescales') for m in self._models]
            self._its_samples = np.zeros((len(timescales_samples), len(self._lags), self.nits))

            for i, ts in enumerate(timescales_samples):
                if ts is not None:
                    ts = ts[:,:self.nits]
                    self._its_samples[:, i, :ts.shape[1]] = ts.T  # copy into array. Leave 0 if there is no timescales

            if np.any(np.isnan(self._its_samples)):
                computed_all = False

        if not computed_all:
            self.logger.warn('Some timescales could not be computed. Timescales array is smaller than '
                             'expected or contains NaNs')


    @property
    def lagtimes(self):
        r"""Return the list of lag times for which timescales were computed.

        """
        return self.lags

    @property
    def lags(self):
        r"""Return the list of lag times for which timescales were computed.

        """
        return self._lags[self._successful_lag_indexes]

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
            return self._its[self._successful_lag_indexes, :]
        else:
            return self._its[self._successful_lag_indexes, process]

    @property
    def samples_available(self):
        r"""Returns True if samples are available and thus sample
        means, standard errors and confidence intervals can be
        obtained
        
        """
        return self._its_samples is not None

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
            return np.mean(self._its_samples[self._successful_lag_indexes, :, :], axis=2)
        else:
            return np.mean(self._its_samples[self._successful_lag_indexes, process, :], axis=1)

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
            return np.std(self._its_samples[self._successful_lag_indexes, :, :], axis=2)
        else:
            return np.std(self._its_samples[self._successful_lag_indexes, process, :], axis=1)

    def get_sample_conf(self, conf=0.95, process=None):
        r"""Returns the confidence interval that contains alpha % of the sample data
        
        Use:
        conf = 0.6827 for 1-sigma confidence interval
        conf = 0.9545 for 2-sigma confidence interval
        conf = 0.9973 for 3-sigma confidence interval
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
            return confidence_interval(self._its_samples[self._successful_lag_indexes, :, :], conf=conf)
        else:
            return confidence_interval(self._its_samples[self._successful_lag_indexes, :, process], conf=conf)

    @property
    def estimators(self):
        r"""Returns the estimators for all lagtimes .

        """
        return [self._estimators[i] for i in self._successful_lag_indexes]

    @property
    def models(self):
        r"""Returns the models for all lagtimes .

        """
        return [self._models[i] for i in self._successful_lag_indexes]

    @property
    def fraction_of_frames(self):
        r"""Returns the fraction of frames used to compute the count matrix at each lagtime.

        Notes
        -------
        In a list of discrete trajectories with varying lengths, the estimation at longer lagtimes will mean
        discarding some trajectories for which not even one count can be computed. This function returns the fraction
        of frames that was actually used in computing the count matrix.

        **Be aware**: this fraction refers to the **full count matrix**, and not that of the largest connected
        set. Hence, the output is not necessarily the **active** fraction. For that, use the
        :py:func:`EstimatedMSM.active_count_fraction` function of the :py:class:`EstimatedMSM` class object.
        """

        # TODO : implement fraction_of_active_frames

        # Are we computing this for the first time?
        if not hasattr(self,'_fraction'):
            self._fraction = np.zeros_like(self.lagtimes, dtype='float32')
            self._nframes = self._trajlengths.sum()

            # Iterate over lagtimes and find trajectories that contributed with at least one count
            for ii, lag in enumerate(self.lagtimes):
                long_enough = np.argwhere(self._trajlengths-lag >= 1).squeeze()
                self._fraction[ii] = self._trajlengths[long_enough].sum()/self._nframes

        return self._fraction