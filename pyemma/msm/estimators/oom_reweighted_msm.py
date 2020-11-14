
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

import warnings

import numpy as _np
from msmtools import estimation as msmest

from pyemma.msm.estimators._OOM_MSM import bootstrapping_count_matrix, bootstrapping_dtrajs, twostep_count_matrix, \
    rank_decision, oom_components, equilibrium_transition_matrix
from pyemma.msm.estimators._msm_estimator_base import _MSMEstimator
from pyemma.util.annotators import fix_docs, aliased


__author__ = 'Feliks Nueske, Fabian Paul'


@fix_docs
@aliased
class OOMReweightedMSM(_MSMEstimator):
    r"""OOM based estimator for MSMs given discrete trajectory statistics"""
    __serialize_version = 0
    __serialize_fields = ('_C2t', '_C_active', '_C_full', '_Xi',
                          '_active_set', '_connected_sets',
                          '_eigenvalues_OOM', '_full2_active',
                          '_is_estimated', '_nstates', '_nstates_full',
                          '_omega', '_sigma', '_oom_rank', '_rank_ind')

    def __init__(self, lag=1, reversible=True, count_mode='sliding', sparse=False, connectivity='largest',
                 dt_traj='1 step', nbs=10000, rank_Ct='bootstrap_counts', tol_rank=10.0,
                 score_method='VAMP2', score_k=10,
                 mincount_connectivity='1/n'):
        r"""Maximum likelihood estimator for MSMs given discrete trajectory statistics

        Parameters
        ----------
        lag : int
            lag time at which transitions are counted and the transition matrix is
            estimated.

        reversible : bool, optional, default = True
            If true compute reversible MSM, else non-reversible MSM

        count_mode : str, optional, default='sliding'
            mode to obtain count matrices from discrete trajectories. Should be
            one of:

            * 'sliding' : A trajectory of length T will have :math:`T-tau` counts
              at time indexes

              .. math::

                 (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)
            * 'sample' : A trajectory of length T will have :math:`T/\tau` counts
              at time indexes

              .. math::

                    (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., (((T/\tau)-1) \tau \rightarrow T)

        sparse : bool, optional, default = False
            If true compute count matrix, transition matrix and all derived
            quantities using sparse matrix algebra. In this case python sparse
            matrices will be returned by the corresponding functions instead of
            numpy arrays. This behavior is suggested for very large numbers of
            states (e.g. > 4000) because it is likely to be much more efficient.
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

        nbs : int, optional, default=10000
            number of re-samplings for rank decision in OOM estimation.

        rank_Ct : str, optional
            Re-sampling method for model rank selection. Can be
            * 'bootstrap_counts': Directly re-sample transitions based on effective count matrix.

            * 'bootstrap_trajs': Re-draw complete trajectories with replacement.

        tol_rank: float, optional, default = 10.0
            signal-to-noise threshold for rank decision.

        score_method : str, optional, default='VAMP2'
            Score to be used with score function. Available are:

            |  'VAMP1'  [1]_
            |  'VAMP2'  [1]_
            |  'VAMPE'  [1]_

        score_k : int or None
            The maximum number of eigenvalues or singular values used in the
            score. If set to None, all available eigenvalues will be used.

        mincount_connectivity : float or '1/n'
            minimum number of counts to consider a connection between two states.
            Counts lower than that will count zero in the connectivity check and
            may thus separate the resulting transition matrix. The default
            evaluates to 1/nstates.

        References
        ----------
        .. [1] H. Wu and F. Noe: Variational approach for learning Markov processes from time series data
            (in preparation)

        """
        # Check count mode:
        self.count_mode = str(count_mode).lower()
        if self.count_mode not in ('sliding', 'sample'):
            raise ValueError('count mode ' + count_mode + ' is unknown. Only \'sliding\' and \'sample\' are allowed.')
        if rank_Ct not in ('bootstrap_counts', 'bootstrap_trajs'):
            raise ValueError('rank_Ct must be either \'bootstrap_counts\' or \'bootstrap_trajs\'')

        super(OOMReweightedMSM, self).__init__(lag=lag, reversible=reversible, count_mode=count_mode, sparse=sparse,
                                               connectivity=connectivity, dt_traj=dt_traj,
                                               score_method=score_method, score_k=score_k,
                                               mincount_connectivity=mincount_connectivity)
        self.nbs = nbs
        self.tol_rank = tol_rank
        self.rank_Ct = rank_Ct

    def _estimate(self, dtrajs):
        """ Estimate MSM """

        if self.core_set is not None:
            raise NotImplementedError('Core set MSMs currently not compatible with {}.'.format(self.__class__.__name__))

        # remove last lag steps from dtrajs:
        dtrajs_lag = [traj[:-self.lag] for traj in dtrajs]

        # get trajectory counts. This sets _C_full and _nstates_full
        dtrajstats = self._get_dtraj_stats(dtrajs_lag)
        self._C_full = dtrajstats.count_matrix()  # full count matrix
        self._nstates_full = self._C_full.shape[0]  # number of states

        # set active set. This is at the same time a mapping from active to full
        if self.connectivity == 'largest':
            self.active_set = dtrajstats.largest_connected_set
        else:
            raise NotImplementedError('OOM based MSM estimation is only implemented for connectivity=\'largest\'.')

        # FIXME: setting is_estimated before so that we can start using the parameters just set, but this is not clean!
        # is estimated
        self._is_estimated = True

        # if active set is empty, we can't do anything.
        if _np.size(self.active_set) == 0:
            raise RuntimeError('Active set is empty. Cannot estimate MSM.')

        # active count matrix and number of states
        self._C_active = dtrajstats.count_matrix(subset=self.active_set)
        self._nstates = self._C_active.shape[0]

        # computed derived quantities
        # back-mapping from full to lcs
        self._full2active = -1 * _np.ones(dtrajstats.nstates, dtype=int)
        self._full2active[self.active_set] = _np.arange(len(self.active_set))

        # Estimate transition matrix
        if self.connectivity == 'largest':
            # Re-sampling:
            if self.rank_Ct == 'bootstrap_counts':
                Ceff_full = msmest.effective_count_matrix(dtrajs_lag, self.lag)
                from pyemma.util.linalg import submatrix
                Ceff = submatrix(Ceff_full, self.active_set)
                smean, sdev = bootstrapping_count_matrix(Ceff, nbs=self.nbs)
            else:
                smean, sdev = bootstrapping_dtrajs(dtrajs_lag, self.lag, self._nstates_full, nbs=self.nbs,
                                                   active_set=self._active_set)
            # Estimate two step count matrices:
            C2t = twostep_count_matrix(dtrajs, self.lag, self._nstates_full)
            # Rank decision:
            rank_ind = rank_decision(smean, sdev, tol=self.tol_rank)
            # Estimate OOM components:
            Xi, omega, sigma, l = oom_components(self._C_full.toarray(), C2t, rank_ind=rank_ind,
                                                 lcc=self.active_set)
            # Compute transition matrix:
            P, lcc_new = equilibrium_transition_matrix(Xi, omega, sigma, reversible=self.reversible)
        else:
            raise NotImplementedError('OOM based MSM estimation is only implemented for connectivity=\'largest\'.')

        # Update active set and derived quantities:
        if lcc_new.size < self._nstates:
            self._active_set = self._active_set[lcc_new]
            self._C_active = dtrajstats.count_matrix(subset=self.active_set)
            self._nstates = self._C_active.shape[0]
            self._full2active = -1 * _np.ones(dtrajstats.nstates, dtype=int)
            self._full2active[self.active_set] = _np.arange(len(self.active_set))
            warnings.warn("Caution: Re-estimation of count matrix resulted in reduction of the active set.")

        # continue sparse or dense?
        if not self.sparse:
            # converting count matrices to arrays. As a result the
            # transition matrix and all subsequent properties will be
            # computed using dense arrays and dense matrix algebra.
            self._C_full = self._C_full.toarray()
            self._C_active = self._C_active.toarray()

        # Done. We set our own model parameters, so this estimator is
        # equal to the estimated model.
        self._dtrajs_full = dtrajs
        self._connected_sets = msmest.connected_sets(self._C_full)
        self._Xi = Xi
        self._omega = omega
        self._sigma = sigma
        self._eigenvalues_OOM = l
        self._rank_ind = rank_ind
        self._oom_rank = self._sigma.size
        self._C2t = C2t
        self.set_model_params(P=P, pi=None, reversible=self.reversible,
                              dt_model=self.timestep_traj.get_scaled(self.lag))

        return self

    def _blocksplit_dtrajs(self, dtrajs, sliding):
        """ Override splitting method of base class.

        For OOM estimators we currently need a clean trajectory splitting, i.e. we don't do block splitting at all.

        """
        if len(dtrajs) < 2:
            raise NotImplementedError('Current cross-validation implementation for OOMReweightedMSM requires' +
                                      'multiple trajectories. You can split the trajectory yourself into training' +
                                      'and test set and use the score method after fitting the training set.')
        return dtrajs

    @property
    def eigenvalues_OOM(self):
        """
            System eigenvalues estimated by OOM.

        """
        self._check_is_estimated()
        return self._eigenvalues_OOM

    @property
    def timescales_OOM(self):
        """
            System timescales estimated by OOM.

        """
        self._check_is_estimated()
        return -self.lag / _np.log(_np.abs(self._eigenvalues_OOM[1:]))

    @property
    def OOM_rank(self):
        """
            Return OOM model rank.

        """
        self._check_is_estimated()
        return self._oom_rank

    @property
    def OOM_components(self):
        """
            Return OOM components.

        """
        self._check_is_estimated()
        return self._Xi

    @property
    def OOM_omega(self):
        """
            Return OOM initial state vector.

        """
        self._check_is_estimated()
        return self._omega

    @property
    def OOM_sigma(self):
        """
            Return OOM evaluator vector.

        """
        self._check_is_estimated()
        return self._sigma
