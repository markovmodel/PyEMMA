
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

from __future__ import absolute_import
from six.moves import range
__author__ = 'noe'

import numpy as np
from pyemma.msm.estimators.estimated_hmsm import EstimatedHMSM as _EstimatedHMSM

from pyemma.msm.estimators.estimated_msm import EstimatedMSM as _EstimatedMSM
from pyemma.msm.estimators.maximum_likelihood_msm import MaximumLikelihoodMSM as _MSMEstimator
from pyemma._base.estimator import Estimator as _Estimator
from pyemma.util import types as _types
from pyemma.util.units import TimeUnit


def _lag_observations(observations, lag, stride=1):
    r""" Create new trajectories that are subsampled at lag but shifted

    Given a trajectory (s0, s1, s2, s3, s4, ...) and lag 3, this function will generate 3 trajectories
    (s0, s3, s6, ...), (s1, s4, s7, ...) and (s2, s5, s8, ...). Use this function in order to parametrize a MLE
    at lag times larger than 1 without discarding data. Do not use this function for Bayesian estimators, where
    data must be given such that subsequent transitions are uncorrelated.

    Parameters
    ----------
    observations : list of int arrays
        observation trajectories
    lag : int
        lag time
    stride : int, default=1
        will return only one trajectory for every stride. Use this for Bayesian analysis.

    """
    obsnew = []
    for obs in observations:
        for shift in range(0, lag, stride):
            obsnew.append(obs[shift:][::lag])
    return obsnew


# TODO: This parameter was in the docstring but is not used:
#     store_data : bool
#         True: estimate() returns an :class:`pyemma.msm.EstimatedMSM` object
#         with discrete trajectories and counts stored. False: estimate() returns
#         a plain :class:`pyemma.msm.MSM` object that only contains the
#         transition matrix and quantities derived from it.


class MaximumLikelihoodHMSM(_Estimator, _EstimatedHMSM):
    r"""Maximum likelihood estimator for a Hidden MSM given a MSM"""

    def __init__(self, nstates=2, lag=1, stride=1, msm_init=None, reversible=True, connectivity='largest',
                 observe_active=True, dt_traj='1 step', accuracy=1e-3, maxit=1000):
        r"""Maximum likelihood estimator for a Hidden MSM given a MSM

        Parameters
        ----------
        nstates : int, optional, default=2
            number of hidden states
        lag : int, optional, default=1
            lagtime to estimate the HMSM at
        stride : str or int, default=1
            stride between two lagged trajectories extracted from the input
            trajectories. Given trajectory s[t], stride and lag will result
            in trajectories
                s[0], s[lag], s[2 lag], ...
                s[stride], s[stride + lag], s[stride + 2 lag], ...
            Setting stride = 1 will result in using all data (useful for maximum
            likelihood estimator), while a Bayesian estimator requires a longer
            stride in order to have statistically uncorrelated trajectories.
            Setting stride = 'effective' uses the largest neglected timescale as
            an estimate for the correlation time and sets the stride accordingly
        msm_init : :class:`MSM <pyemma.msm.estimators.msm_estimated.MSM>`
            MSM object to initialize the estimation
        reversible : bool, optional, default = True
            If true compute reversible MSM, else non-reversible MSM
        connectivity : str, optional, default = 'largest'
            Connectivity mode. Three methods are intended (currently only 'largest' is implemented)
            * 'largest' : The active set is the largest reversibly connected set. All estimation will be done on this
              subset and all quantities (transition matrix, stationary distribution, etc) are only defined on this
              subset and are correspondingly smaller than the full set of states
            * 'all' : The active set is the full set of states. Estimation will be conducted on each reversibly
              connected set separately. That means the transition matrix will decompose into disconnected
              submatrices, the stationary vector is only defined within subsets, etc. Currently not implemented.
            * 'none' : The active set is the full set of states. Estimation will be conducted on the full set of
              states without ensuring connectivity. This only permits nonreversible estimation. Currently not
              implemented.
        observe_active : bool, optional, default=True
            True: Restricts the observation set to the active states of the MSM.
            False: All states are in the observation set.
        dt_traj : str, optional, default='1 step'
            Description of the physical time corresponding to the trajectory time
            step.  May be used by analysis algorithms such as plotting tools to
            pretty-print the axes. By default '1 step', i.e. there is no physical
            time unit. Specify by a number, whitespace and unit. Permitted units
            are (* is an arbitrary string):

            |  'fs',  'femtosecond*'
            |  'ps',  'picosecond*'
            |  'ns',  'nanosecond*'
            |  'us',  'microsecond*'
            |  'ms',  'millisecond*'
            |  's',   'second*'

        accuracy : float, optional, default = 1e-3
            convergence threshold for EM iteration. When two the likelihood does
            not increase by more than accuracy, the iteration is stopped
            successfully.
        maxit : int, optional, default = 1000
            stopping criterion for EM iteration. When so many iterations are
            performed without reaching the requested accuracy, the iteration is
            stopped without convergence (a warning is given)

        """
        self.nstates = nstates
        self.lag = lag
        self.stride = stride
        self.msm_init = msm_init
        self.reversible = reversible
        self.connectivity = connectivity
        self.observe_active = observe_active
        self.dt_traj = dt_traj
        self.timestep_traj = TimeUnit(dt_traj)
        self.accuracy = accuracy
        self.maxit = maxit


    #TODO: store_data is mentioned but not implemented or used!
    def _estimate(self, dtrajs):
        """

        Parameters
        ----------

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.estimators.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        # ensure right format
        dtrajs = _types.ensure_dtraj_list(dtrajs)
        # if no initial MSM is given, estimate it now
        if self.msm_init is None:
            # estimate with sparse=False, because we need to do PCCA which is currently not implemented for sparse
            # estimate with store_data=True, because we need an EstimatedMSM
            msm_estimator = _MSMEstimator(lag=self.lag, reversible=self.reversible, sparse=False,
                                          connectivity=self.connectivity, dt_traj=self.timestep_traj)
            msm_init = msm_estimator.estimate(dtrajs)
        else:
            assert isinstance(self.msm_init, _EstimatedMSM), 'msm_init must be of type EstimatedMSM'
            msm_init = self.msm_init
            self.reversible = msm_init.is_reversible

        # print 'Connected set: ', msm_init.active_set

        # generate lagged observations
        if self.stride == 'effective':
            # by default use lag as stride (=lag sampling), because we currently have no better theory for deciding
            # how many uncorrelated counts we can make
            self.stride = self.lag
            # if we have more than nstates timescales in our MSM, we use the next (neglected) timescale as an
            # estimate of the decorrelation time
            if msm_init.nstates > self.nstates:
                corrtime = int(max(1, msm_init.timescales()[self.nstates-1]))
                # use the smaller of these two pessimistic estimates
                self.stride = min(self.stride, 2*corrtime)
        # TODO: Here we always use the full observation state space for the estimation.
        dtrajs_lagged = _lag_observations(dtrajs, self.lag, stride=self.stride)

        # check input
        assert _types.is_int(self.nstates) and self.nstates > 1 and self.nstates <= msm_init.nstates, \
            'nstates must be an int in [2,msmobj.nstates]'
        # if hmm.nstates = msm.nstates there is no problem. Otherwise, check spectral gap
        if msm_init.nstates > self.nstates:
            timescale_ratios = msm_init.timescales()[:-1] / msm_init.timescales()[1:]
            if timescale_ratios[self.nstates-2] < 2.0:
                self.logger.warning('Requested coarse-grained model with ' + str(self.nstates) + ' metastable states at ' +
                                 'lag=' + str(self.lag) + '.' + 'The ratio of relaxation timescales between ' +
                                 str(self.nstates) + ' and ' + str(self.nstates+1) + ' states is only ' +
                                 str(timescale_ratios[self.nstates-2]) + ' while we recommend at least 2. ' +
                                 ' It is possible that the resulting HMM is inaccurate. Handle with caution.')

        # set things from MSM
        # TODO: dtrajs_obs is set here, but not used in estimation. Estimation is alwas done with
        # TODO: respect to full observation (see above). This is confusing. Define how we want to do this in gen.
        # TODO: observable set is also not used, it is just saved.
        nstates_obs_full = msm_init.nstates_full
        if self.observe_active:
            nstates_obs = msm_init.nstates
            observable_set = msm_init.active_set
            dtrajs_obs = msm_init.discrete_trajectories_active
        else:
            nstates_obs = msm_init.nstates_full
            observable_set = np.arange(nstates_obs_full)
            dtrajs_obs = msm_init.discrete_trajectories_full

        # TODO: this is redundant with BHMM code because that code is currently not easily accessible and
        # TODO: we don't want to re-estimate. Should be reengineered in bhmm.
        # ---------------------------------------------------------------------------------------
        # PCCA-based coarse-graining
        # ---------------------------------------------------------------------------------------
        # pcca- to number of metastable states
        pcca = msm_init.pcca(self.nstates)

        # HMM output matrix
        eps = 0.01 * (1.0/nstates_obs_full)  # default output probability, in order to avoid zero columns
        # Use PCCA distributions, but at least eps to avoid 100% assignment to any state (breaks convergence)
        B_conn = np.maximum(msm_init.metastable_distributions, eps)
        # full state space output matrix
        B = eps * np.ones((self.nstates, nstates_obs_full), dtype=np.float64)
        # expand B_conn to full state space
        # TODO: here we always select the active set, no matter if observe_active=True or False.
        B[:, msm_init.active_set] = B_conn[:, :]
        # TODO: at this point we will have zero observation probabilities for states that are not in the active
        # TODO: set. If these occur in the trajectory, that will mean zero columns in the output probabilities
        # TODO: and crash of forward-backward and sampling algorithms.
        # renormalize B to make it row-stochastic
        B /= B.sum(axis=1)[:, None]

        # coarse-grained transition matrix
        P_coarse = pcca.coarse_grained_transition_matrix
        # take care of unphysical values. First symmetrize
        X = np.dot(np.diag(pcca.coarse_grained_stationary_probability), P_coarse)
        X = 0.5*(X + X.T)
        # if there are values < 0, set to eps
        X = np.maximum(X, eps)
        # turn into coarse-grained transition matrix
        A = X / X.sum(axis=1)[:, None]

        # ---------------------------------------------------------------------------------------
        # Estimate discrete HMM
        # ---------------------------------------------------------------------------------------
        # lazy import bhmm here in order to avoid dependency loops
        import bhmm
        # initialize discrete HMM
        hmm_init = bhmm.discrete_hmm(A, B, stationary=True, reversible=self.reversible)
        # run EM
        hmm = bhmm.estimate_hmm(dtrajs_lagged, self.nstates, lag=1, initial_model=hmm_init,
                                accuracy=self.accuracy, maxit=self.maxit)
        self.hmm = bhmm.DiscreteHMM(hmm)

        # find observable set
        transition_matrix = self.hmm.transition_matrix
        observation_probabilities = self.hmm.output_probabilities
        # TODO: Cutting down... OK, this can be done
        if self.observe_active:  # cut down observation probabilities to active set
            observation_probabilities = observation_probabilities[:, msm_init.active_set]
            observation_probabilities /= observation_probabilities.sum(axis=1)[:,None]  # renormalize

        # parametrize self
        self._dtrajs_full = dtrajs
        self._dtrajs_lagged = dtrajs_lagged
        self._observable_set = observable_set
        self._dtrajs_obs = dtrajs_obs
        self.set_model_params(P=transition_matrix, pobs=observation_probabilities,
                              reversible=self.reversible, dt_model=self.timestep_traj.get_scaled(self.lag))

        return self

    def cktest(self, mlags=10, conf=0.95, err_est=False, show_progress=True):
        """ Conducts a Chapman-Kolmogorow test.

        Parameters
        ----------
        mlags : int or int-array, default=10
            multiples of lag times for testing the Model, e.g. range(10).
            A single int will trigger a range, i.e. mlags=10 maps to
            mlags=range(10). The setting None will choose mlags automatically
            according to the longest available trajectory
        conf : float, optional, default = 0.95
            confidence interval
        err_est : bool, default=False
            compute errors also for all estimations (computationally expensive)
            If False, only the prediction will get error bars, which is often
            sufficient to validate a model.
        show_progress : bool, default=True
            Show progressbars for calculation?


        References
        ----------
        This is an adaption of the Chapman-Kolmogorov Test described in detail
        in [1]_ to Hidden MSMs as described in [2]_.

        .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
            Chodera, C Schuette and F Noe. 2011. Markov models of
            molecular kinetics: Generation and validation. J Chem Phys
            134: 174105

        .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner: Projected and hidden
            Markov models for calculating kinetics and metastable states of complex
            molecules. J. Chem. Phys. 139, 184114 (2013)

        """
        from pyemma.msm.estimators import ChapmanKolmogorovValidator
        ck = ChapmanKolmogorovValidator(self, self, np.eye(self.nstates),
                                        mlags=mlags, conf=conf, err_est=err_est,
                                        show_progress=show_progress)
        ck.estimate(self._dtrajs_full)
        return ck