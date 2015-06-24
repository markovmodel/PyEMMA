__author__ = 'noe'

from pyemma._base.estimator import Estimator as _Estimator
from pyemma.msm.estimators.maximum_likelihood_msm import MaximumLikelihoodMSM as _MSMEstimator
from pyemma.msm.models.msm_estimated import EstimatedMSM as _EstimatedMSM
from pyemma.msm.models.msm_sampled import SampledMSM as _SampledMSM


class BayesianMSM(_Estimator):
    """ Bayesian estimator for MSMs given discrete trajectory statistics

    Parameters
    ----------
    lag : int, optional, default=1
        lagtime to estimate the HMSM at
    nsample : int, optional, default=1000
        number of sampled transition matrices used
    nstep : int, optional, default=None
        number of Gibbs sampling steps for each transition matrix used.
        If None, nstep will be determined automatically
    msm : :class:`MSM <pyemma.msm.models.MSM>`
        Single-point estimate of MSM object around which errors will be evaluated
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

    """
    def __init__(self, lag=1, nsample=1000, nstep=None, init_msm=None, reversible=True, sparse=False,
                 connectivity='largest', dt='1 step', conf=0.683):
        self.lag = lag
        self.nsample = nsample
        self.nstep = nstep
        self.init_msm = init_msm
        self.reversible = reversible
        self.sparse = sparse
        self.connectivity = connectivity
        self.dt = dt
        self.conf = conf

    def _estimate(self, dtrajs):
        """

        Parameters
        ----------
        dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
            discrete trajectories, stored as integer ndarrays (arbitrary size)
            or a single ndarray for only one trajectory.

        Return
        ------
        hmsm : :class:`EstimatedHMSM <pyemma.msm.ui.hmsm_estimated.EstimatedHMSM>`
            Estimated Hidden Markov state model

        """
        # if no initial MSM is given, estimate it now
        if self.init_msm is None:
            msm_estimator = _MSMEstimator(lag=self.lag, reversible=self.reversible, sparse=self.sparse,
                                          connectivity=self.connectivity, dt=self.dt)
            init_msm = msm_estimator.estimate(dtrajs)
        else:  # override constructor settings when meaningful
            # check input
            assert isinstance(self.init_msm, _EstimatedMSM), 'msm must be of type EstimatedMSM'
            self.reversible = self.init_msm.is_reversible
            init_msm = self.init_msm

        # transition matrix sampler
        from pyemma.msm.estimation import tmatrix_sampler
        from math import sqrt
        if self.nstep is None:
            self.nstep = int(sqrt(init_msm.nstates))  # heuristic for number of steps to decorrelate
        tsampler = tmatrix_sampler(init_msm.count_matrix_active, reversible=self.reversible, nstep=self.nstep)
        sample_Ps, sample_mus = tsampler.sample(nsample=self.nsample, return_statdist=True,
                                                T_init=init_msm.transition_matrix)
        # construct MSM
        sampled_msm = _SampledMSM(init_msm, sample_Ps, sample_mus, conf=self.conf)
        return sampled_msm
