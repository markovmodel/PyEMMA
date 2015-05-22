__author__ = 'noe'

import numpy as np

from pyemma.msm import estimation as msmest
from pyemma.util.annotators import shortcut
from pyemma.util.log import getLogger
from pyemma.msm.ui.dtraj_stats import DiscreteTrajectoryStats as _DiscreteTrajectoryStats
from pyemma.msm.ui.msm_estimator import MSMEstimator as _MSMEstimator
from pyemma.msm.ui.msm_estimated import EstimatedMSM as _EstimatedMSM
from pyemma.msm.ui.msm_sampled import SampledMSM as _SampledMSM


class BayesianMSMEstimator:
    """ Maximum likelihood estimator for MSMs given discrete trajectory statistics

    """
    def __init__(self, dtrajs, reversible=True, sparse=False, connectivity='largest', dt='1 step', conf=0.683, **kwargs):
        """
            Parameters
            ----------
            dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
                discrete trajectories, stored as integer ndarrays (arbitrary size)
                or a single ndarray for only one trajectory.
            reversible : bool, optional, default = True
                If true compute reversible MSM, else non-reversible MSM
            sparse : bool, optional, default = False
                If true compute count matrix, transition matrix and all derived quantities using sparse matrix algebra.
                In this case python sparse matrices will be returned by the corresponding functions instead of numpy
                arrays. This behavior is suggested for very large numbers of states (e.g. > 4000) because it is likely
                to be much more efficient.
            connectivity : str, optional, default = 'largest'
                Connectivity mode. Three methods are intended (currently only 'largest' is implemented)
                'largest' : The active set is the largest reversibly connected set. All estimation will be done on this
                    subset and all quantities (transition matrix, stationary distribution, etc) are only defined on this
                    subset and are correspondingly smaller than the full set of states
                'all' : The active set is the full set of states. Estimation will be conducted on each reversibly connected
                    set separately. That means the transition matrix will decompose into disconnected submatrices,
                    the stationary vector is only defined within subsets, etc. Currently not implemented.
                'none' : The active set is the full set of states. Estimation will be conducted on the full set of states
                    without ensuring connectivity. This only permits nonreversible estimation. Currently not implemented.
            dt : str, optional, default='1 step'
                Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
                plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
                Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):

                |  'fs',  'femtosecond*'
                |  'ps',  'picosecond*'
                |  'ns',  'nanosecond*'
                |  'us',  'microsecond*'
                |  'ms',  'millisecond*'
                |  's',   'second*'

            **kwargs: Optional algorithm-specific parameters. See below for special cases
            maxiter = 1000000 : int
                Optional parameter with reversible = True.
                maximum number of iterations before the transition matrix estimation method exits
            maxerr = 1e-8 : float
                Optional parameter with reversible = True.
                convergence tolerance for transition matrix estimation.
                This specifies the maximum change of the Euclidean norm of relative
                stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
                :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
                probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.

        """
        # MSM estimation parameters
        self._dtrajs = dtrajs
        self._reversible = reversible
        self._msm_sparse = sparse
        self._msm_connectivity = connectivity
        self._dt = dt
        self._kwargs = kwargs
        # BMSM estimation parameters
        self._conf = conf
        # run estimation
        self._estimated = False


    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    def estimate(self, lag=1, nsample=1000, nstep=None, msm=None):
        """

        Parameters
        ----------
        lag : int, optional, default=1
            lagtime to estimate the HMSM at
        nsample : int, optional, default=1000
            number of sampled transition matrices used
        nstep : int, optional, default=None
            number of Gibbs sampling steps for each transition matrix used.
            If None, nstep will be determined automatically
        msm : :class:`HMSM <pyemma.msm.ui.hmsm.HMSM>`
            Single-point estimate of HMSM object around which errors will be evaluated

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.ui.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        # set lag time
        self._lag = lag

        # if no initial MSM is given, estimate it now
        if msm is None:
            msm_estimator = _MSMEstimator(self._dtrajs, reversible=self._reversible, sparse=self._msm_sparse,
                                            connectivity=self._msm_connectivity, dt=self._dt, **self._kwargs)
            msm = msm_estimator.estimate(lag=lag)
        else:  # override constructor settings when meaningful
            # check input
            assert isinstance(msm, _EstimatedMSM), 'msm must be of type EstimatedMSM'

        # transition matrix sampler
        from pyemma.msm.estimation import tmatrix_sampler
        from math import sqrt
        if nstep is None:
            nstep = int(sqrt(msm.nstates))  # heuristic for number of steps to decorrelate
        tsampler = tmatrix_sampler(msm.count_matrix_active, reversible=msm.is_reversible, nstep=nstep)
        sample_Ps, sample_mus = tsampler.sample(nsample=nsample, return_statdist=True, T_init=msm.transition_matrix)
        # construct MSM
        sampled_msm = _SampledMSM(msm, sample_Ps, sample_mus, conf=self._conf)
        return sampled_msm
