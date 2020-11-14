
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2019 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import numpy as _np
from msmtools import estimation as msmest

from pyemma.msm.estimators import MaximumLikelihoodMSM
from pyemma.util.annotators import fix_docs, aliased
from pyemma.util.statistics import confidence_interval as _ci


__all__ = ['AugmentedMarkovModel']
__author__ = 'Simon Olsson'

@fix_docs
@aliased
class AugmentedMarkovModel(MaximumLikelihoodMSM):
    r"""AMM estimator given discrete trajectory statistics and stationary expectation values from experiments"""

    __serialize_version = 0
    __serialize_fields = ('E_active', 'E_min', 'E_max', 'mhat', 'm', 'lagrange',
                          'sigmas', 'count_inside', 'count_outside')

    def __init__(self, lag=1, count_mode='sliding', connectivity='largest',
                 dt_traj='1 step',
                 E=None, m=None, w=None, eps=0.05, support_ci=1.00, maxiter=500, max_cache=3000,
                 mincount_connectivity='1/n', core_set=None, milestoning_method='last_core'):

        r"""Maximum likelihood estimator for AMMs given discrete trajectory statistics and expectation values from experiments

        Parameters
        ----------
        lag : int
            lag time at which transitions are counted and the transition matrix is
            estimated.

        count_mode : str, optional, default='sliding'
            mode to obtain count matrices from discrete trajectories. Should be
            one of:

            * 'sliding' : A trajectory of length T will have :math:`T-tau` counts
              at time indexes

              .. math::

                 (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)
            * 'sample' : A trajectory of length T will have :math:`T/tau` counts
              at time indexes

              .. math::

                    (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/tau)-1) \tau \rightarrow T)

        connectivity : str, optional, default = 'largest'
            Connectivity mode. Three methods are intended (currently only 'largest'
            is implemented)

            * 'largest' : The active set is the largest reversibly connected set.
              All estimation will be done on this subset and all quantities
              (transition matrix, stationary distribution, etc) are only defined
              on this subset and are correspondingly smaller than the full set
              of states
            * 'all' : The active set is the full set of states. Estimation will be
              conducted on each reversibly connected set separately. That means
              the transition matrix will decompose into disconnected submatrices,
              the stationary vector is only defined within subsets, etc.
              Currently not implemented.
            * 'none' : The active set is the full set of states. Estimation will
              be conducted on the full set of
              states without ensuring connectivity. This only permits
              nonreversible estimation. Currently not implemented.

        dt_traj : str, optional, default='1 step'
            Description of the physical time of the input trajectories. May be used
            by analysis algorithms such as plotting tools to pretty-print the axes.
            By default '1 step', i.e. there is no physical time unit. Specify by a
            number, whitespace and unit. Permitted units are (* is an arbitrary
            string):

            |  'fs',  'femtosecond*'
            |  'ps',  'picosecond*'
            |  'ns',  'nanosecond*'
            |  'us',  'microsecond*'
            |  'ms',  'millisecond*'
            |  's',   'second*'

        E : ndarray(n, k)
          Expectations by state. n Markov states, k experimental observables; each index is average over members of the Markov state.

        m : ndarray(k)
          Experimental measurements.

        w : ndarray(k)
          Weights of experimental measurement (1/2s^2), where s is the std error.

        eps : float, default=0.05
          Convergence criterion for Lagrange multipliers. When the relative error on the observable average becomes less than this value for all observables, convergence is reached.

        support_ci : float, default=1.00
          Confidence interval for determination whether experimental data are inside or outside Markov model support

        maxiter : int, default=500
          Maximum number of iterations

        max_cache : int, default=3000
          Maximum size (in megabytes) of cache when computing R tensor (Supporting information in [1]).

        mincount_connectivity : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. The default
            evaluates to 1/nstates.


        References
        ----------
        .. [1] Olsson S, Wu H, Paul F, Clementi C, Noe F: Combining experimental and simulation data of molecular
            processes via augmented Markov models. PNAS 114, 8265-8270 (2017).
        """
        # Check count mode:
        self.count_mode = str(count_mode).lower()
        if self.count_mode not in ('sliding', 'sample'):
            raise ValueError('count mode ' + count_mode + ' is unknown. Only \'sliding\' and \'sample\' are allowed.')

        super(AugmentedMarkovModel, self).__init__(lag=lag, reversible=True, count_mode=count_mode, sparse=False,
                                                   connectivity=connectivity, dt_traj=dt_traj, score_method=None,
                                                   score_k=None, mincount_connectivity=mincount_connectivity,
                                                   maxiter=maxiter, core_set=core_set,
                                                   milestoning_method=milestoning_method)

        self.E = E
        if E is not None:
            self.n_mstates, self.n_exp = _np.shape(E)
        else:
            self.logger.info("AugmentedMarkovModel instance initialized without Markov state observable matrix (E). "
                             "This is necessary for estimation.")
        if m is None:
            self.logger.info("AugmentedMarkovModel instance initialized without experimental values (m). "
                             "This necessary for estimation.")
        if w is None:
            self.logger.info("AugmentedMarkovModel instance initialized without experimental weights (w). "
                             "This is necessary for estimation.")

        self.m = m
        self.w = w

        # Relative error for lagrange convergence assessment.
        self.eps = eps

        # Specifies the confidence interval of experimental values consider inside or outside support of the simulation
        # Is used to identify experimental data which have values never visited in the simulation, user is informed about these,
        # and lagrange estimation for these stops when the slope reaches a (near) constant value.
        self.support_ci = support_ci

        # check for zero weights
        if w is not None:
            if _np.any(w < 1e-12):
                raise ValueError("Some weights are close to zero or negative. Please remove these from input.")
            # compute uncertainties
            self.sigmas = _np.sqrt(1. / 2. / self.w)
        else:
            self.sigmas = None

        # Convergence flag for pihat
        self.max_cache = max_cache

    @staticmethod
    def _log_likelihood_biased(C, T, E, mhat, ws):
        """ Evaluate AMM likelihood. """
        ll_unbiased = msmest.log_likelihood(C, T)
        ll_bias = -_np.sum(ws * (mhat - E) ** 2.)
        return ll_unbiased + ll_bias

    def _update_G(self):
        """ Update G.
            Observable covariance.
            See SI of [1].
        """
        self._G = (_np.dot(self.E_active.T, self.E_active * self._pihat[:, None]) -
                   self.mhat[:, None] * self.mhat[None, :])

    def _update_Q(self):
        """ Compute Q, a weighted sum of the R-tensor.

            See SI of [1].
        """
        self._Q = _np.zeros((self.n_mstates_active, self.n_mstates_active))
        for k in range(self.n_exp_active):
            self._Q = self._Q + self.w[k] * self._S[k] * self._get_Rk(k)
        self._Q *= -2.

    def _update_Rslices(self, i):
        """ Computation of multiple slices of R tensor.

            When _estimate(.) is called the R-tensor is split into segments whose maximum size is
            specified by max_cache argument (see constructor).
            _Rsi specifies which of the segments are currently in cache.
             For equations check SI of [1].

        """
        pek = self._pihat[:, None] * self.E_active[:, i * self._slicesz:(i + 1) * self._slicesz]
        pp = self._pihat[:, None] + self._pihat[None, :]
        ppmhat = pp * self.mhat[i * self._slicesz:(i + 1) * self._slicesz, None, None]
        self._Rs = (pek[:, None, :] + pek[None, :, :]).T - ppmhat
        self._Rsi = i

    def _get_Rk(self, k):
        """
          Convienence function to get cached value of an Rk slice of the R tensor.
          If we are outside cache, update the cache and return appropriate slice.

        """
        if k > (self._Rsi + 1) * self._slicesz or k < self._Rsi * self._slicesz:
            self._update_Rslices(_np.floor(k / self._slicesz).astype(int))
            return self._Rs[k % self._slicesz]
        else:
            return self._Rs[k % self._slicesz]

    def _update_pihat(self):
        r""" Update stationary distribution estimate of Augmented Markov model (\hat pi) """
        expons = _np.einsum('i,ji->j', self.lagrange, self.E_active)
        # expons = (self.lagrange[:, None]*self.E_active.T).sum(axis=0)
        expons = expons - expons.max()

        _ph_unnom = self.pi * _np.exp(expons)
        self._pihat = (_ph_unnom / _ph_unnom.sum()).reshape(-1, )

    def _update_mhat(self):
        """ Updates mhat (expectation of observable of the Augmented Markov model) """
        self.mhat = self._pihat.dot(self.E_active)
        self._update_S()

    def _update_S(self):
        """ Computes slope in observable space """
        self._S = self.mhat - self.m

    def _update_X_and_pi(self):
        # evaluate count-over-pi
        c_over_pi = self._csum / self.pi
        D = c_over_pi[:, None] + c_over_pi + self._Q
        # update estimate
        self.X = self._C2 / D

        # renormalize
        self.X /= _np.sum(self.X)
        self.pi = _np.sum(self.X, axis=1)

    def _newton_lagrange(self):
        """
          This function performs a Newton update of the Lagrange multipliers.
          The iteration is constrained by strictly improving the AMM likelihood, and yielding meaningful stationary properties.

          TODO: clean up and optimize code.
        """
        # initialize a number of values
        l_old = self.lagrange.copy()
        _ll_new = -_np.inf
        frac = 1.
        mhat_old = self.mhat.copy()
        while self._ll_old > _ll_new or _np.any(self._pihat < 1e-12):
            self._update_pihat()
            self._update_G()
            # Lagrange slope calculation
            dl = 2. * (frac * self._G * self.w[:, None] * self._S[:, None]).sum(axis=0)
            # update Lagrange multipliers
            self.lagrange = l_old - frac * dl
            self._update_pihat()
            # a number of sanity checks
            while _np.any(self._pihat < 1e-12) and frac > 0.05:
                frac = frac * 0.5
                self.lagrange = l_old - frac * dl
                self._update_pihat()

            self.lagrange = l_old - frac * dl
            self._update_pihat()
            self._update_mhat()
            self._update_Q()
            self._update_X_and_pi()

            P = self.X / self.pi[:, None]
            _ll_new = self._log_likelihood_biased(self._C_active, P, self.m, self.mhat, self.w)
            # decrease slope in Lagrange space (only used if loop is repeated, e.g. if sanity checks fail)
            frac *= 0.1

            if frac < 1e-12:
                self.logger.info("Small gradient fraction")
                break

            self._dmhat = self.mhat - mhat_old
            self._ll_old = float(_ll_new)

        self._lls.append(_ll_new)

    def _estimate(self, dtrajs):
        if self.E is None or self.w is None or self.m is None:
            raise ValueError("E, w or m was not specified. Stopping.")

        # get trajectory counts. This sets _C_full and _nstates_full
        dtrajstats = self._get_dtraj_stats(dtrajs)
        self._C_full = dtrajstats.count_matrix()  # full count matrix
        self._nstates_full = self._C_full.shape[0]  # number of states

        # set active set. This is at the same time a mapping from active to full
        if self.connectivity == 'largest':
            # statdist not given - full connectivity on all states
            self.active_set = dtrajstats.largest_connected_set
        else:
            # for 'None' and 'all' all visited states are active
            self.active_set = dtrajstats.visited_set

        # FIXME: setting is_estimated before so that we can start using the parameters just set, but this is not clean!
        # is estimated
        self._is_estimated = True

        # if active set is empty, we can't do anything.
        if _np.size(self.active_set) == 0:
            raise RuntimeError('Active set is empty. Cannot estimate AMM.')

        # active count matrix and number of states
        self._C_active = dtrajstats.count_matrix(subset=self.active_set)
        self._nstates = self._C_active.shape[0]

        # computed derived quantities
        # back-mapping from full to lcs
        self._full2active = -1 * _np.ones(dtrajstats.nstates, dtype=int)
        self._full2active[self.active_set] = _np.arange(len(self.active_set))

        # slice out active states from E matrix

        _dset = list(set(_np.concatenate(self._dtrajs_full)))
        _rras = [_dset.index(s) for s in self.active_set]
        self.E_active = self.E[_rras]

        if not self.sparse:
            self._C_active = self._C_active.toarray()
            self._C_full = self._C_full.toarray()

        # reversibly counted
        self._C2 = 0.5 * (self._C_active + self._C_active.T)
        self._nz = _np.nonzero(self._C2)
        self._csum = _np.sum(self._C_active, axis=1)  # row sums C

        # get ranges of Markov model expectation values
        if self.support_ci == 1:
            self.E_min = _np.min(self.E_active, axis=0)
            self.E_max = _np.max(self.E_active, axis=0)
        else:
            # PyEMMA confidence interval calculation fails sometimes with conf=1.0
            self.E_min, self.E_max = _ci(self.E_active, conf=self.support_ci)

        # dimensions of E matrix
        self.n_mstates_active, self.n_exp_active = _np.shape(self.E_active)

        assert self.n_exp_active == len(self.w)
        assert self.n_exp_active == len(self.m)

        self.count_outside = []
        self.count_inside = []
        self._lls = []

        i = 0
        # Determine which experimental values are outside the support as defined by the Confidence interval
        for emi, ema, mm, mw in zip(self.E_min, self.E_max, self.m, self.w):
            if mm < emi or ema < mm:
                self.logger.info("Experimental value %f is outside the support (%f,%f)" % (mm, emi, ema))
                self.count_outside.append(i)
            else:
                self.count_inside.append(i)
            i = i + 1

        self.logger.info(
            "Total experimental constraints outside support %d of %d" % (len(self.count_outside), len(self.E_min)))

        # A number of initializations
        self.P, self.pi = msmest.tmatrix(self._C_active, reversible=True, return_statdist=True)
        self.lagrange = _np.zeros(self.m.shape)
        self._pihat = self.pi.copy()
        self._update_mhat()
        self._dmhat = 1e-1 * _np.ones(_np.shape(self.mhat))

        # Determine number of slices of R-tensors computable at once with the given cache size
        self._slicesz = _np.floor(self.max_cache / (self.P.nbytes / 1.e6)).astype(int)
        # compute first bundle of slices
        self._update_Rslices(0)

        self._ll_old = self._log_likelihood_biased(self._C_active, self.P, self.m, self.mhat, self.w)

        self._lls = [self._ll_old]

        # make sure everything is initialized

        self._update_pihat()
        self._update_mhat()

        self._update_Q()
        self._update_X_and_pi()

        self._ll_old = self._log_likelihood_biased(self._C_active, self.P, self.m, self.mhat, self.w)
        self._update_G()

        #
        # Main estimation algorithm
        # 2-step algorithm, lagrange multipliers and pihat have different convergence criteria
        # when the lagrange multipliers have converged, pihat is updated until the log-likelihood has converged (changes are smaller than 1e-3).
        # These do not always converge together, but usually within a few steps of each other.
        # A better heuristic for the latter may be necessary. For realistic cases (the two ubiquitin examples in [1])
        # this yielded results very similar to those with more stringent convergence criteria (changes smaller than 1e-9) with convergence times
        # which are seconds instead of tens of minutes.
        #

        converged = False  # Convergence flag for lagrange multipliers
        i = 0
        die = False
        while i <= self.maxiter:
            pihat_old = self._pihat.copy()
            self._update_pihat()
            if not _np.all(self._pihat > 0):
                self._pihat = pihat_old.copy()
                die = True
                self.logger.warning("pihat does not have a finite probability for all states, terminating")
            self._update_mhat()
            self._update_Q()
            if i > 1:
                X_old = self.X.copy()
                self._update_X_and_pi()
                if _np.any(self.X[self._nz] < 0) and i > 0:
                    die = True
                    self.logger.warning(
                        "Warning: new X is not proportional to C... reverting to previous step and terminating")
                    self.X = X_old.copy()

            if not converged:
                self._newton_lagrange()
            else:  # once Lagrange multipliers are converged compute likelihood here
                P = self.X / self.pi[:, None]
                _ll_new = self._log_likelihood_biased(self._C_active, P, self.m, self.mhat, self.w)
                self._lls.append(_ll_new)

            # General case fixed-point iteration
            if len(self.count_outside) > 0:
                if i > 1 and _np.all((_np.abs(self._dmhat) / self.sigmas) < self.eps) and not converged:
                    self.logger.info("Converged Lagrange multipliers after %i steps..." % i)
                    converged = True
            # Special case
            else:
                if _np.abs(self._lls[-2] - self._lls[-1]) < 1e-8:
                    self.logger.info("Converged Lagrange multipliers after %i steps..." % i)
                    converged = True
            # if Lagrange multipliers are converged, check whether log-likelihood has converged
            if converged and _np.abs(self._lls[-2] - self._lls[-1]) < 1e-8:
                self.logger.info("Converged pihat after %i steps..." % i)
                die = True
            if die:
                break
            if i == self.maxiter:
                self.logger.info("Failed to converge within %i iterations. "
                                 "Consider increasing max_iter(now=%i)" % (i, self.max_iter))
            i += 1

        _P = msmest.tmatrix(self._C_active, reversible=True, mu=self._pihat)

        self._connected_sets = msmest.connected_sets(self._C_full)
        self.set_model_params(P=_P, pi=self._pihat, reversible=True,
                              dt_model=self.timestep_traj.get_scaled(self.lag))

        return self

    def hmm(self, n):
        self.logger.info("Not Implemented - Please use PCCA for now.")

    def score(self, dtrajs, score_method=None, score_k=None):
        self.logger.info("Not Implemented.")
