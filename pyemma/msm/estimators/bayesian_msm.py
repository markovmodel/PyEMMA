__author__ = 'noe'

from pyemma.msm.models.msm import MSM as _MSM
from pyemma.msm.estimators.maximum_likelihood_msm import MaximumLikelihoodMSM as _MLMSM
from pyemma.msm.models.msm_sampled import SampledMSM as _SampledMSM
from pyemma.util.types import ensure_dtraj_list

class BayesianMSM(_MLMSM, _SampledMSM):
    """ Bayesian estimator for MSMs given discrete trajectory statistics

    Parameters
    ----------
    lag : int, optional, default=1
        lagtime to estimate the HMSM at

    nsamples : int, optional, default=100
        number of sampled transition matrices used

    nstep : int, optional, default=None
        number of Gibbs sampling steps for each transition matrix used.
        If None, nstep will be determined automatically

    msm : :class:`MSM <pyemma.msm.models.MSM>`
        Single-point estimate of MSM object around which errors will be
        evaluated

    reversible : bool, optional, default = True
        If true compute reversible MSM, else non-reversible MSM

    sparse : bool, optional, default = False
        If true compute count matrix, transition matrix and all derived
        quantities using sparse matrix algebra. In this case python sparse
        matrices will be returned by the corresponding functions instead of
        numpy arrays. This behavior is suggested for very large numbers of
        states (e.g. > 4000) because it is likely to be much more efficient.

    connectivity : str, optional, default = 'largest'
        Connectivity mode. Three methods are intended (currently only
        'largest' is implemented)
        'largest' : The active set is the largest reversibly connected set.
            All estimation will be done on this subset and all quantities
            (transition matrix, stationary distribution, etc) are only defined
            on this subset and are correspondingly smaller than the full set
            of states
        'all' : The active set is the full set of states. Estimation will be
            conducted on each reversibly connected set separately. That means
            the transition matrix will decompose into disconnected submatrices,
            the stationary vector is only defined within subsets, etc.
            Currently not implemented.
        'none' : The active set is the full set of states. Estimation will be
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

    """
    def __init__(self, lag=1, nsamples=100, nstep=None, reversible=True, sparse=False,
                 connectivity='largest', dt_traj='1 step', conf=0.95):
        _MLMSM.__init__(self, lag=lag, reversible=reversible, sparse=sparse, connectivity=connectivity, dt_traj=dt_traj)
        self.nsamples = nsamples
        self.nstep = nstep
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
        # ensure right format
        dtrajs = ensure_dtraj_list(dtrajs)
        # conduct MLE estimation (superclass) first
        _MLMSM._estimate(self, dtrajs)

        # transition matrix sampler
        from pyemma.msm.estimation import tmatrix_sampler
        from math import sqrt
        if self.nstep is None:
            self.nstep = int(sqrt(self.nstates))  # heuristic for number of steps to decorrelate
        tsampler = tmatrix_sampler(self.effective_count_matrix, reversible=self.reversible, nstep=self.nstep)
        sample_Ps, sample_mus = tsampler.sample(nsample=self.nsamples, return_statdist=True,
                                                T_init=self.transition_matrix)
        # construct sampled MSMs
        samples = []
        for i in range(self.nsamples):
            samples.append(_MSM(sample_Ps[i], pi=sample_mus[i], reversible=self.reversible, dt_model=self.dt_model))

        # update self model
        self.update_model_params(samples=samples)

        # done
        return self
