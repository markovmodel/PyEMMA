
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Jul 26, 2014

@author: noe
'''

from __future__ import absolute_import, print_function

import numpy as np

from pyemma._base.parallel import NJobsMixIn
from pyemma.util.annotators import estimation_required, alias, aliased

from pyemma.util.statistics import confidence_interval
from pyemma.util import types as _types
from pyemma._base.estimator import Estimator, get_estimator, param_grid, estimate_param_scan
from pyemma._base.progress import ProgressReporter
from pyemma._base.model import SampledModel

__docformat__ = "restructuredtext en"

__all__ = ['ImpliedTimescales']

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
    import decimal
    while lag <= maxlag:
        lag = lag*multiplier
        # round up, like python 2
        lag = int(decimal.Decimal(lag).quantize(decimal.Decimal('1'),    
                                                rounding=decimal.ROUND_HALF_UP))
        if lag <= maxlag:
            ilag = int(lag)
            lags.append(ilag)
    return np.array(lags)


def _hash_dtrajs(dtraj_list):
    x = 0
    from pyemma.util.numeric import _hash_numpy_array
    for d in dtraj_list:
        x ^= _hash_numpy_array(d)
    return x


# TODO: build a generic implied timescales estimate in _base, and make this a subclass (for dtrajs)
# TODO: Timescales should be assigned by similar eigenvectors rather than by order
# TODO: when requesting too long lagtimes, throw a warning and exclude lagtime from calculation, but compute the rest
@aliased
class ImpliedTimescales(Estimator, ProgressReporter, NJobsMixIn):
    r"""Implied timescales for a series of lag times.

    Parameters
    ----------
    estimator : Estimator
        Estimator to be used for estimating timescales at each lag time.

    lags : array-like with integers or None, optional
        integer lag times at which the implied timescales will be calculated. If set to None (default)
        as list of lagtimes will be automatically generated.

    nits : int, optional
        maximum number of implied timescales to be computed and stored. If less
        timescales are available, nits will be set to a smaller value during
        estimation. None means the number of timescales will be automatically
        determined.

    n_jobs: int, optional
        how many subprocesses to start to estimate the models for each lag time.

    """
    def __init__(self, estimator, lags=None, nits=None, n_jobs=1,
                 show_progress=True):
        # initialize
        self.estimator = get_estimator(estimator)
        self.nits = nits
        self.n_jobs = n_jobs
        self.show_progress = show_progress

        self._lag_hist = []
        self.lags = lags

        # estimated its. 2D-array with indexing: lagtime, its
        self._its = None
        # sampled its's. 3D-array with indexing: lagtime, its, sample
        self._its_samples = None

    def estimate(self, X, **params):
        """
        Parameters
        ----------
        X: lists of integer arrays
            discrete trajectories

        estimator : Estimator
            Estimator to be used for estimating timescales at each lag time.

        lags : array-like with integers or None, optional
            integer lag times at which the implied timescales will be calculated. If set to None (default)
            as list of lagtimes will be automatically generated.

        nits : int, optional
            maximum number of implied timescales to be computed and stored. If less
            timescales are available, nits will be set to a smaller value during
            estimation. None means the number of timescales will be automatically
            determined.

        n_jobs: int, optional
            how many subprocesses to start to estimate the models for each lag time.
        """
        return super(ImpliedTimescales, self).estimate(X, **params)

    def _estimate(self, dtrajs):
        ### PREPARE AND CHECK DATA
        # TODO: Currently only discrete trajectories are implemented. For a general class this needs to be changed.
        dtrajs = _types.ensure_dtraj_list(dtrajs)

        # check trajectory lengths
        if self._estimated:
            # if dtrajs has now changed, unset the _estimated flag to re-set every derived quantity.
            assert hasattr(self, '_last_dtrajs_input_hash')
            if self._last_dtrajs_input_hash != _hash_dtrajs(dtrajs):
                self.logger.warning("estimating from new data, discard all previously computed models.")
                self._estimated = False
        else:
            self._last_dtrajs_input_hash = _hash_dtrajs(dtrajs)

        self._trajlengths = np.fromiter((len(traj) for traj in dtrajs), dtype=int, count=len(dtrajs))
        maxlength = np.max(self._trajlengths)

        # set lag times by data if not yet set
        if self._lags is None:
            maxlag = 0.5 * np.sum(self._trajlengths) / float(len(self._trajlengths))
            self._lags = _generate_lags(maxlag, 1.5)

        # check if some lag times are forbidden.
        if np.max(self._lags) >= maxlength:
            Ifit = np.where(self._lags < maxlength)[0]
            Inofit = np.where(self._lags >= maxlength)[0]
            self.logger.warning('Ignoring lag times that exceed the longest trajectory: %s', self._lags[Inofit])
            self._lags = self._lags[Ifit]

        ### RUN ESTIMATION
        if self._estimated:
            # we already had run an estimation, determine which lag times we need to compute
            # TODO: this will re-evaluate problematic lag times, wont it?
            lags = sorted(list(set(self._lags).difference(self._last_lags)))
            if len(lags) == 0:
                self.logger.info("All lag times already estimated.")
                return self
            assert lags
            self.logger.info("Running estimating for not yet estimated lags times: %s", lags)
        else:
            lags = self._lags

        # construct all parameter sets for the estimator
        param_sets = tuple(param_grid({'lag': lags}))

        if isinstance(self.estimator, SampledModel):
            self.estimator.show_progress = False

        # run estimation on all lag times
        models, estimators = estimate_param_scan(self.estimator, dtrajs, param_sets, failfast=False,
                                                 return_estimators=True, n_jobs=self.n_jobs,
                                                 progress_reporter=self)
        self._estimators = estimators

        self._postprocess_results(models)

    def _postprocess_results(self, models):
        ### PROCESS RESULTS
        # if some results are None, estimation has failed. Warn and truncate models and lag times
        good = np.array([i for i, m in enumerate(models) if m is not None], dtype=int)
        bad = np.array([i for i, m in enumerate(models) if m is None], dtype=int)
        if good.size == 0:
            raise RuntimeError('Estimation has failed at ALL lagtimes. Check for errors.')
        if bad.size > 0:
            self.logger.warning('Estimation has failed at lagtimes: {lags}. '
                                'Run single-lag estimation at these lags to track down the error.',
                                lags=self._lags[bad])
            models = list(np.array(models)[good])

        # merge models prior evaluation
        if self._estimated:
            # determine new input order for new models
            assert all(hasattr(m, 'lag') for m in models)
            # sorted by lag time
            models = sorted(self._models + models, key=lambda m: m.lag)

        self._models = models
        # timescales
        timescales = [m.timescales() for m in models]

        # how many finite timescales do we really have?
        maxnts = max([len(ts[np.isfinite(ts)]) for ts in timescales])
        if self.nits is None:
            self.nits = maxnts

        if maxnts < self.nits:
            self.nits = maxnts
            self.logger.warning('Changed user setting nits to the number of available timescales nits=%s', self.nits)

        # sort timescales into matrix
        computed_all = True  # flag if we have found any problems
        self._its = np.empty((len(self._lags), self.nits))
        self._its[:] = np.NAN  # initialize with NaN in order to point out timescales that were not computed
        self._successful_lag_indexes = []
        for i, ts in enumerate(timescales):
            if ts is not None and np.any(np.isfinite(ts)):  # if there are any finite timescales available, add them
                self._its[i, :len(ts)] = ts[:self.nits]  # copy into array. Leave NaN if there is no timescale
                self._successful_lag_indexes.append(i)

        self._successful_lag_indexes = np.array(self._successful_lag_indexes)
        if len(self._successful_lag_indexes) < len(self._lags):
            computed_all = False
        if np.any(np.isnan(self._its)):
            computed_all = False

        # timescales samples if available
        if isinstance(models[0], SampledModel):
            # samples
            timescales_samples = [m.sample_f('timescales') for m in models]
            nsamples = np.shape(timescales_samples[0])[0]
            self._its_samples = np.empty((nsamples, len(self._lags), self.nits))
            self._its_samples[:] = np.NAN  # initialize with NaN in order to point out timescales that were not computed

            for i, ts in enumerate(timescales_samples):
                if ts is not None:
                    ts = np.vstack(ts)
                    ts = ts[:, :self.nits]
                    self._its_samples[:, i, :ts.shape[1]] = ts  # copy into array. Leave NaN if there is no timescales

            if np.any(np.isnan(self._its_samples)):
                computed_all = False

        if not computed_all:
            self.logger.warning('Some timescales could not be computed. Timescales array is smaller than '
                                'expected or contains NaNs')

    @property
    @alias('lagtimes')
    def lags(self):
        """Return the list of lag times for which timescales were computed."""
        return self._lags[self._successful_lag_indexes]

    @lags.setter
    def lags(self, lags):
        """Sets the lag times at which the models will be estimated.
        Remembers the last non None value, in order to extend the lag list. If the input data during estimation is unchanged, we will only
        need to re-estimate new lag times. If a lag time is removed, we clean up the underlying models and
        derived data (timescales).
        """
        if hasattr(self, '_lags') and self._lags is not None:
            self._last_lags = frozenset(self._lags)
            # remove obsolete models and computed data.
            if self._models and lags is not None:
                survivors = np.array([i for i in self._successful_lag_indexes if self._models[i].lag in lags])
                if survivors.size == 0:
                    self._models = []
                    self._its = None
                    self._its_samples = None
                    self._successful_lag_indexes = None
                else:
                    self._models = np.array(self._models)[survivors].tolist()
                    self._successful_lag_indexes = np.arange(len(self._models))
                    self._its = self._its[survivors]
                    if self.samples_available:
                        self._its_samples = self._its_samples[survivors]
        else:
            self._last_lags = set()

        # set lag times
        if _types.is_int(lags):  # got a single integer. We create a list
            self._lags = _generate_lags(lags, 1.5)
        elif lags is None: # obtain a series of meaningful lag times from the input trajectory length
            self._lags = None
        else:  # got a list of ints or None - otherwise raise exception.
            self._lags = _types.ensure_int_vector_or_None(lags, require_order=True)
            self._lags.sort()

    @property
    @alias('number_of_timescales')
    def nits(self):
        """Return the number of timescales."""
        return self._nits

    @nits.setter
    def nits(self, value):
        self._nits = int(value) if value is not None else None

    @property
    @estimation_required
    def timescales(self):
        r"""Returns the implied timescale estimates

        Returns
        -------
        timescales : ndarray((l x k), dtype=float)
            timescales for all processes and lag times.
            l is the number of lag times and k is the number of computed timescales.

        """
        return self.get_timescales()

    def get_timescales(self, process=None):
        r"""Returns the implied timescale estimates

        Parameters
        ----------
        process : int or None, default = None
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
        r"""Returns the sample means of implied timescales. Only available if underlying estimator produces samples.

        Parameters
        ----------
        process : int or None, default = None
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
            raise RuntimeError('Cannot compute sample mean, because no samples were generated. '
                               'Use an estimator, which provides samples')
        # OK, go:
        if process is None:
            return np.mean(self._its_samples[:, self._successful_lag_indexes, :], axis=0)
        else:
            return np.mean(self._its_samples[:, self._successful_lag_indexes, process], axis=0)

    @property
    def sample_std(self):
        r"""Returns the standard error of implied timescales. Only available if underlying estimator produces samples.


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
        r"""Returns the standard error of implied timescales. Only available if underlying estimator produces samples.

        Parameters
        ----------
        process : int or None, default = None
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
            raise RuntimeError('Cannot compute sample std, because no samples were generated. '
                               'Use an estimator, which provides samples')
        # OK, go:
        if process is None:
            return np.std(self._its_samples[:, self._successful_lag_indexes, :], axis=0)
        else:
            return np.std(self._its_samples[:, self._successful_lag_indexes, process], axis=0)

    def get_sample_conf(self, conf=0.95, process=None):
        r"""Returns the confidence interval that contains alpha % of the sample data


        etc.

        Parameters
        ----------
        conf : float, default = 0.95
            the confidence interval. Use:

            * conf = 0.6827 for 1-sigma confidence interval
            * conf = 0.9545 for 2-sigma confidence interval
            * conf = 0.9973 for 3-sigma confidence interval

        Returns
        -------
        (L,R) : (float[],float[]) or (float[][],float[][])
            lower and upper timescales bounding the confidence interval

            * if process is None, will return two (l x k) arrays, where l is the number of lag times
              and k is the number of computed timescales.
            * if process is an integer, will return two (l)-arrays with the
              selected process time scale for every lag time

        """
        if self._its_samples is None:
            raise RuntimeError('Cannot compute sample conf, because no samples were generated. '
                               'Use an estimator, which provides samples')
        # OK, go:
        if process is None:
            return confidence_interval(self._its_samples[:, self._successful_lag_indexes, :], conf=conf)
        else:
            return confidence_interval(self._its_samples[:, self._successful_lag_indexes, process], conf=conf)

    @property
    def estimators(self):
        r"""Returns the estimators for all lagtimes.

        """
        return [self._estimators[i] for i in self._successful_lag_indexes]

    @property
    def models(self):
        r"""Returns the models for all lagtimes.

        """
        return [self._models[i] for i in self._successful_lag_indexes]

    @property
    def fraction_of_frames(self):
        r"""Returns the fraction of frames used to compute the count matrix at each lag time.
        Notes
        -----
        In a list of discrete trajectories with varying lengths, the estimation at longer lag times will mean
        discarding some trajectories for which not even one count can be computed. This function returns the fraction
        of frames that was actually used in computing the count matrix.

        **Be aware**: this fraction refers to the **full count matrix**, and not that of the largest connected
        set. Hence, the output is not necessarily the **active** fraction. For that, use the
        :py:meth:`activte_count_fraction <pyemma.msm.MaximumLikelihoodMSM.active_count_fraction>` function of
        the :py:class:`pyemma.msm.MaximumLikelihoodMSM` class object or for HMM respectively.
        """
        # TODO : implement fraction_of_active_frames
        # Are we computing this for the first time?
        if not hasattr(self, '_fraction'):
            self._fraction = np.zeros_like(self.lagtimes, dtype='float32')
            self._nframes = self._trajlengths.sum()

            # Iterate over lagtimes and find trajectories that contributed with at least one count
            for ii, lag in enumerate(self.lagtimes):
                long_enough = np.argwhere(self._trajlengths - lag >= 1).squeeze()
                self._fraction[ii] = self._trajlengths[long_enough].sum() / self._nframes

        return self._fraction
