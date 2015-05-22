__author__ = 'noe'

import numpy as np

from pyemma.msm import estimation as msmest
from pyemma.util.annotators import shortcut
from pyemma.util.log import getLogger
from pyemma.msm.ui.dtraj_stats import DiscreteTrajectoryStats as _DiscreteTrajectoryStats
from pyemma.msm.ui.msm_estimated import EstimatedMSM as _EstimatedMSM


# TODO: We can probably get rid of most functions, because we produce an EstimatedMSM object that has all the info.
# TODO: Only keep estimation-specific information, such as likelihood convergence info.
class MSMEstimator:
    """ Maximum likelihood estimator for MSMs given discrete trajectory statistics

    """
    def __init__(self, dtrajs, reversible=True, sparse=False, connectivity='largest', dt='1 step', **kwargs):
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
        # set basic parameters
        self._dtrajstats = _DiscreteTrajectoryStats(dtrajs)
        self._reversible = reversible

        # sparse matrix computation wanted?
        self._sparse = sparse
        if sparse:
            self._logger.warn('Sparse mode is currently untested and might lead to errors. '
                               'I strongly suggest to use sparse=False unless you know what you are doing.')
        if self._dtrajstats.nstates > 4000 and not sparse:
            self._logger.warn('Building a dense MSM with ' + str(self._dtrajstats.nstates) + ' states. This can be '
                              'inefficient or unfeasible in terms of both runtime and memory consumption. '
                              'Consider using sparse=True.')

        # store connectivity mode (lowercase)
        self._connectivity = connectivity.lower()
        if self._connectivity == 'largest':
            pass  # this is the current default. no need to do anything
        elif self._connectivity == 'all':
            raise NotImplementedError('MSM estimation with connectivity=\'all\' is currently not implemented.')
        elif self._connectivity == 'none':
            raise NotImplementedError('MSM estimation with connectivity=\'none\' is currently not implemented.')
        else:
            raise ValueError('connectivity mode ' + str(connectivity) + ' is unknown.')

        # time step
        self._dt = dt

        # additional transition matrix parameters
        self._kwargs = kwargs

        # run estimation
        self._estimated = False


    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    def estimate(self, lag=1):
        # set lag time
        self._lag = lag

        # count lagged
        self._dtrajstats.count_lagged(self._lag)

        # set active set
        if self._connectivity == 'largest':
            # the largest connected set is the active set. This is at the same time a mapping from active to full
            self._active_set = self._dtrajstats.largest_connected_set
        else:
            # for 'None' and 'all' all visited states are active
            self._active_set = self._dtrajstats.visited_set

        # number of states in active set
        self._nstates = len(self._active_set)

        # count matrices
        self._C_full = self._dtrajstats.count_matrix()
        self._C_active = self._dtrajstats.count_matrix(subset=self._active_set)

        # computed derived quantities
        # back-mapping from full to lcs
        self._full2active = -1 * np.ones((self._dtrajstats.nstates), dtype=int)
        self._full2active[self._active_set] = np.array(range(len(self._active_set)), dtype=int)

        # continue sparse or dense?
        if not self._sparse:
            # converting count matrices to arrays. As a result the transition matrix and all subsequent properties
            # will be computed using dense arrays and dense matrix algebra.
            self._C_full = self._C_full.toarray()
            self._C_active = self._C_active.toarray()

        # Estimate transition matrix
        if self._connectivity == 'largest':
            self._T = msmest.transition_matrix(self._C_active, reversible=self._reversible, **self._kwargs)
        elif self._connectivity == 'none':
            # reversible mode only possible if active set is connected - in this case all visited states are connected
            # and thus this mode is identical to 'largest'
            if self._reversible and not msmest.is_connected(self._C_active):
                raise ValueError('Reversible MSM estimation is not possible with connectivity mode \'none\', '+
                                 'because the set of all visited states is not reversibly connected')
            self._T = msmest.transition_matrix(self._C_active, reversible=self._reversible, **self._kwargs)
        else:
            raise NotImplementedError(
                'MSM estimation with connectivity=\'self.connectivity\' is currently not implemented.')

        # construct MSM
        self._msm = _EstimatedMSM(self._dtrajstats.discrete_trajectories, self._dt, self._lag, self._connectivity,
                                  self._active_set, self._dtrajstats.connected_sets, self._C_full, self._C_active,
                                  self._T)

        # check consistency of estimated transition matrix
        if self._reversible != self._msm.is_reversible:
            self._logger.warn('Reversible was set but transition matrix did not pass reversibility check. Check your '
                              'settings. If they are alright this might just be a small numerical deviation from a '
                              'truly reversible matrix. Will carefully continue with nonreversible transition matrix.')

        self._estimated = True
        return self._msm

    def _assert_estimated(self):
        assert self._estimated, "MSM hasn't been estimated yet, make sure to call estimate()"

    @property
    def estimated(self):
        """Returns whether this msm has been estimated yet"""
        return self._estimated

    @property
    def nstates_full(self):
        """
        The number of all states in the discrete trajectories

        """
        return self._dtrajstats.nstates

    @property
    def is_reversible(self):
        """Returns whether the MSM is reversible """
        return self._reversible

    @property
    def is_sparse(self):
        """Returns whether the MSM is sparse """
        return self._sparse

    @property
    def connectivity(self):
        """Returns the connectivity mode of the MSM """
        return self._connectivity

    @property
    def dt(self):
        """Returns the time step"""
        return self._dt

    @property
    def lagtime(self):
        """ The lag time in steps """
        self._assert_estimated()
        return self._lag

    @property
    def nstates(self):
        """
        The number of states on which all computations and estimations will be done

        """
        self._assert_estimated()
        return self._nstates

    @property
    @shortcut('dtrajs')
    def discrete_trajectories(self):
        """
        A list of integer arrays with the discrete trajectories mapped to the connectivity mode used.
        For example, for connectivity='largest', the indexes will be given within the connected set.
        Frames that are not in the connected set will be -1.

        """
        self._assert_estimated()
        return self._dtrajstats.discrete_trajectories

    @property
    def active_set(self):
        """
        The reversible connected sets of states, sorted by size (descending)

        """
        self._assert_estimated()
        return self._active_set
    @property
    def connected_sets(self):
        """
        The reversible connected sets of states, sorted by size (descending)

        """
        self._assert_estimated()
        return self._dtrajstats.connected_sets

    @property
    @shortcut('dtrajs_active')
    def discrete_trajectories_active(self):
        """
        A list of integer arrays with the discrete trajectories mapped to the connectivity mode used.
        For example, for connectivity='largest', the indexes will be given within the connected set.
        Frames that are not in the connected set will be -1.

        """
        self._assert_estimated()
        # compute connected dtrajs
        self._dtrajs_active = []
        for dtraj in self._dtrajstats.discrete_trajectories:
            self._dtrajs_active.append(self._full2active[dtraj])

        return self._dtrajs_active

    @property
    def count_matrix_active(self):
        """The count matrix on the active set given the connectivity mode used.

        For example, for connectivity='largest', the count matrix is given only on the largest reversibly connected set.
        Attention: This count matrix has been obtained by sliding a window of length tau across the data. It contains
        a factor of tau more counts than are statistically uncorrelated. It's fine to use this matrix for maximum
        likelihood estimated, but it will give far too small errors if you use it for uncertainty calculations. In order
        to do uncertainty calculations, use the effective count matrix, see:
        :meth:`effective_count_matrix`

        See Also
        --------
        effective_count_matrix
            For a count matrix with effective (statistically uncorrelated) counts.

        """
        self._assert_estimated()
        return self._C_active

    @property
    def count_matrix_full(self):
        """
        The count matrix on full set of discrete states, irrespective as to whether they are connected or not.
        Attention: This count matrix has been obtained by sliding a window of length tau across the data. It contains
        a factor of tau more counts than are statistically uncorrelated. It's fine to use this matrix for maximum
        likelihood estimated, but it will give far too small errors if you use it for uncertainty calculations. In order
        to do uncertainty calculations, use the effective count matrix, see: :attribute:`effective_count_matrix`
        (only implemented on the active set), or divide this count matrix by tau.

        See Also
        --------
        effective_count_matrix
            For a active-set count matrix with effective (statistically uncorrelated) counts.

        """
        return self._C_full

    @property
    def transition_matrix(self):
        """
        The transition matrix, estimated on the active set. For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        self._assert_estimated()
        return self._T

    @property
    def active2full(self):
        r""" Mapping from active to full state indexes

        """
        return self._active_set

    @property
    def full2active(self):
        r""" Mapping from full to active state indexes

        """
        return self._full2active
