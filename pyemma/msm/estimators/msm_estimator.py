__author__ = 'noe'

import numpy as np

from pyemma._base.estimator import Estimator
from pyemma.msm import estimation as msmest
from pyemma.msm.util.dtraj_stats import DiscreteTrajectoryStats as _DiscreteTrajectoryStats
from pyemma.msm.models.msm_estimated import EstimatedMSM as _EstimatedMSM


class MSMEstimator(Estimator):
    """ Maximum likelihood estimator for MSMs given discrete trajectory statistics

    """
    def __init__(self, lag=None, reversible=True, sparse=False, connectivity='largest', dt='1 step',
                 maxiter = 1000000, maxerr = 1e-8):
        """
            Parameters
            ----------
            lagtime : int
                lag time at which transitions are counted and the transition matrix is estimated.
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
                'all' : The active set is the full set of states. Estimation will be conducted on each reversibly
                    connected set separately. That means the transition matrix will decompose into disconnected
                    submatrices, the stationary vector is only defined within subsets, etc. Currently not implemented.
                'none' : The active set is the full set of states. Estimation will be conducted on the full set of
                    states without ensuring connectivity. This only permits nonreversible estimation. Currently not
                    implemented.
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
            maxiter = 1000000 : int
                Optional parameter with reversible = True.
                maximum number of iterations before the transition matrix estimation method exits
            maxerr = 1e-8 : float
                Optional parameter with reversible = True.
                convergence tolerance for transition matrix estimation.
                This specifies the maximum change of the Euclidean norm of relative
                stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
                :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in
                small probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.

        """
        self.lag = lag

        # set basic parameters
        self.reversible = reversible

        # sparse matrix computation wanted?
        self.sparse = sparse
        if sparse:
            self.logger.warn('Sparse mode is currently untested and might lead to errors. '
                             'I strongly suggest to use sparse=False unless you know what you are doing.')

        # store connectivity mode (lowercase)
        self.connectivity = connectivity.lower()
        if self.connectivity == 'largest':
            pass  # this is the current default. no need to do anything
        elif self.connectivity == 'all':
            raise NotImplementedError('MSM estimation with connectivity=\'all\' is currently not implemented.')
        elif self.connectivity == 'none':
            raise NotImplementedError('MSM estimation with connectivity=\'none\' is currently not implemented.')
        else:
            raise ValueError('connectivity mode ' + str(connectivity) + ' is unknown.')

        # time step
        self.dt = dt

        # convergence parameters
        self.maxiter = maxiter
        self.maxerr = maxerr

        # run estimation
        self._estimated = False


    def _estimate(self, dtrajs):
        """
            Parameters
            ----------
            dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int) or
            :class:`pyemma.msm.util.dtraj_states.DiscreteTrajectoryStats`
                discrete trajectories, stored as integer ndarrays (arbitrary size)
                or a single ndarray for only one trajectory.
            **params : Other keyword parameters if different from the settings when this estimator was constructed

            Returns
            -------
            MSM : :class:`pyemma.msm.EstimatedMSM` or :class:`pyemma.msm.MSM`

        """
        if isinstance(dtrajs, _DiscreteTrajectoryStats):
            dtrajstats = dtrajs
        else:
            # compute and store discrete trajectory statistics
            dtrajstats = _DiscreteTrajectoryStats(dtrajs)
            # check if this MSM seems too large to be dense
            if dtrajstats.nstates > 4000 and not self.sparse:
                self.logger.warn('Building a dense MSM with ' + str(dtrajstats.nstates) + ' states. This can be '
                                  'inefficient or unfeasible in terms of both runtime and memory consumption. '
                                  'Consider using sparse=True.')

        # count lagged
        dtrajstats.count_lagged(self.lag)

        # set active set
        if self.connectivity == 'largest':
            # the largest connected set is the active set. This is at the same time a mapping from active to full
            active_set = dtrajstats.largest_connected_set
        else:
            # for 'None' and 'all' all visited states are active
            active_set = dtrajstats.visited_set

        # count matrices
        C_full = dtrajstats.count_matrix()
        C_active = dtrajstats.count_matrix(subset=active_set)

        # computed derived quantities
        # back-mapping from full to lcs
        full2active = -1 * np.ones((dtrajstats.nstates), dtype=int)
        full2active[active_set] = np.array(range(len(active_set)), dtype=int)

        # Estimate transition matrix
        if self.connectivity == 'largest':
            T = msmest.transition_matrix(C_active, reversible=self.reversible, maxiter=self.maxiter, maxerr=self.maxerr)
        elif self.connectivity == 'none':
            # reversible mode only possible if active set is connected - in this case all visited states are connected
            # and thus this mode is identical to 'largest'
            if self.reversible and not msmest.is_connected(C_active):
                raise ValueError('Reversible MSM estimation is not possible with connectivity mode \'none\', '+
                                 'because the set of all visited states is not reversibly connected')
            T = msmest.transition_matrix(C_active, reversible=self.reversible, maxiter=self.maxiter, maxerr=self.maxerr)
        else:
            raise NotImplementedError(
                'MSM estimation with connectivity=\'self.connectivity\' is currently not implemented.')

        # continue sparse or dense?
        if not self.sparse:
            # converting count matrices to arrays. As a result the transition matrix and all subsequent properties
            # will be computed using dense arrays and dense matrix algebra.
            C_full = C_full.toarray()
            C_active = C_active.toarray()
            T = T.toarray()

        # construct MSM
        msm = _EstimatedMSM(dtrajstats.discrete_trajectories, self.dt, self.lag, self.connectivity,
                            active_set, dtrajstats.connected_sets, C_full, C_active, T)

        # check consistency of estimated transition matrix
        if self.reversible != self.reversible:
            self._logger.warn('Reversible was set but transition matrix did not pass reversibility check. Check your '
                              'settings. If they are alright this might just be a small numerical deviation from a '
                              'truly reversible matrix. Will carefully continue with nonreversible transition matrix.')

        return msm

