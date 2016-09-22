
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

from pyemma._base.progress import ProgressReporter
from pyemma.msm.estimators.maximum_likelihood_msm import MaximumLikelihoodMSM as _MLMSM
from pyemma.msm.models.msm import MSM as _MSM
from pyemma.msm.models.msm_sampled import SampledMSM as _SampledMSM
from pyemma.util.annotators import fix_docs
from pyemma.util.types import ensure_dtraj_list

__author__ = 'noe'


@fix_docs
class BayesianMSM(_MLMSM, _SampledMSM, ProgressReporter):
    r"""Bayesian Markov state model estimator"""

    def __init__(self, lag=1, nsamples=100, nsteps=None, reversible=True,
                 statdist_constraint=None, count_mode='effective', sparse=False,
                 connectivity='largest', dt_traj='1 step', conf=0.95,
                 show_progress=True):
        r""" Bayesian estimator for MSMs given discrete trajectory statistics

        Parameters
        ----------
        lag : int, optional, default=1
            lagtime to estimate the HMSM at

        nsamples : int, optional, default=100
            number of sampled transition matrices used

        nsteps : int, optional, default=None
            number of Gibbs sampling steps for each transition matrix used.
            If None, nstep will be determined automatically
        
        reversible : bool, optional, default = True
            If true compute reversible MSM, else non-reversible MSM

        statdist_constraint : (M,) ndarray optional
            Stationary vector on the full set of states. Assign zero
            stationary probabilities to states for which the
            stationary vector is unknown. Estimation will be made such
            that the resulting ensemble of transition matrices is
            defined on the intersection of the states with positive
            stationary vector and the largest connected set
            (undirected). 

        count_mode : str, optional, default='effective'
            mode to obtain count matrices from discrete trajectories. Should be one of:

            * 'sliding' : A trajectory of length T will have :math:`T-\tau` counts
              at time indexes
              .. math:: (0 \rightarray \tau), (1 \rightarray \tau+1), ..., (T-\tau-1 \rightarray T-1)

            * 'effective' : Uses an estimate of the transition counts that are
              statistically uncorrelated. Recommended when used with a
              Bayesian MSM.

            * 'sample' : A trajectory of length T will have :math:`T / \tau` counts
              at time indexes
              .. math:: (0 \rightarray \tau), (\tau \rightarray 2 \tau), ..., (((T/tau)-1) \tau \rightarray T)

        sparse : bool, optional, default = False
            If true compute count matrix, transition matrix and all derived
            quantities using sparse matrix algebra. In this case python sparse
            matrices will be returned by the corresponding functions instead of
            numpy arrays. This behavior is suggested for very large numbers of
            states (e.g. > 4000) because it is likely to be much more efficient.

        connectivity : str, optional, default = 'largest'
            Connectivity mode. Three methods are intended (currently only
            'largest' is implemented)

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
            * 'none' : The active set is the full set of states. Estimation will be
              conducted on the full set of states without ensuring connectivity.
              This only permits nonreversible estimation.
              Currently not implemented.

        dt_traj : str, optional, default='1 step'
            Description of the physical time corresponding to the trajectory time
            step. May be used by analysis algorithms such as plotting tools to
            pretty-print the axes. By default '1 step', i.e. there is no physical
            time unit. Specify by a number, whitespace and unit. Permitted units
            are (* is an arbitrary string):

            |  'fs',  'femtosecond*'
            |  'ps',  'picosecond*'
            |  'ns',  'nanosecond*'
            |  'us',  'microsecond*'
            |  'ms',  'millisecond*'
            |  's',   'second*'

        conf : float, optional, default=0.95
            Confidence interval. By default one-sigma (68.3%) is used. Use 95.4%
            for two sigma or 99.7% for three sigma.

        show_progress : bool, default=True
            Show progressbars for calculation?

        References
        ----------
        .. [1] Trendelkamp-Schroer, B., H. Wu, F. Paul and F. Noe: Estimation and
            uncertainty of reversible Markov models. J. Chem. Phys. (in review)
            Preprint: http://arxiv.org/abs/1507.05990

        """
        _MLMSM.__init__(self, lag=lag, reversible=reversible,
                        statdist_constraint=statdist_constraint,
                        count_mode=count_mode, sparse=sparse,
                        connectivity=connectivity, dt_traj=dt_traj)
        self.nsamples = nsamples
        self.nsteps = nsteps
        self.conf = conf
        self.show_progress = show_progress

    def estimate(self, dtrajs, **kw):
        """

        Parameters
        ----------
        dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
            discrete trajectories, stored as integer ndarrays (arbitrary size)
            or a single ndarray for only one trajectory.

        kw : dict
            Other Parameters. See documentation of class.

        Return
        ------
        msm : :class:`BayesianMSM <pyemma.msm.BayesianMSM>`
            Estimated Hidden Markov state model

        """
        return super(BayesianMSM, self).estimate(dtrajs, **kw)

    def _estimate(self, dtrajs):
        # ensure right format
        dtrajs = ensure_dtraj_list(dtrajs)
        # conduct MLE estimation (superclass) first
        _MLMSM._estimate(self, dtrajs)

        # transition matrix sampler
        from msmtools.estimation import tmatrix_sampler
        from math import sqrt
        if self.nsteps is None:
            self.nsteps = int(sqrt(self.nstates))  # heuristic for number of steps to decorrelate
        # use the same count matrix as the MLE. This is why we have effective as a default
        if self.statdist_constraint is None:
            tsampler = tmatrix_sampler(self.count_matrix_active, reversible=self.reversible, T0=self.transition_matrix,
                                       nsteps=self.nsteps)
        else:
            # Use the stationary distribution on the active set of states
            statdist_active = self.pi
            # We can not use the MLE as T0. Use the initialization in the reversible pi sampler
            tsampler = tmatrix_sampler(self.count_matrix_active, reversible=self.reversible,
                                       mu=statdist_active, nsteps=self.nsteps)

        self._progress_register(self.nsamples, description="Sampling MSMs", stage=0)

        if self.show_progress:
            def call_back():
                self._progress_update(1, stage=0)
        else:
            call_back = None

        sample_Ps, sample_mus = tsampler.sample(nsamples=self.nsamples,
                                                return_statdist=True,
                                                call_back=call_back)
        self._progress_force_finish(0)

        # construct sampled MSMs
        samples = []
        for i in range(self.nsamples):
            samples.append(_MSM(sample_Ps[i], pi=sample_mus[i], reversible=self.reversible, dt_model=self.dt_model))

        # update self model
        self.update_model_params(samples=samples)

        # done
        return self
