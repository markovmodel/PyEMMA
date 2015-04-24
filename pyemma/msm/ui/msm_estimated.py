__author__ = 'noe'

import numpy as np

from msm import MSM
from pyemma.util.annotators import shortcut
from pyemma.util.log import getLogger

class EstimatedMSM(MSM):
    r"""Estimates a Markov model from discrete trajectories.

    Parameters
    ----------
    dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
        discrete trajectories, stored as integer ndarrays (arbitrary size)
        or a single ndarray for only one trajectory.
    lag : int
        lagtime for the MSM estimation in multiples of trajectory steps
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
    estimate : bool, optional, default=True
        If true estimate the MSM when creating the MSM object.
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

    Notes
    -----
    You can postpone the estimation of the MSM using estimate=False and
    initiate the estimation procedure by manually calling the MSM.estimate()
    method.

    """

    def __init__(self, dtrajs, lag,
                 reversible=True, sparse=False, connectivity='largest', estimate=True,
                 dt='1 step',
                 **kwargs):
        # TODO: extensive input checking!
        from pyemma.util.types import ensure_dtraj_list

        # start logging
        self.__create_logger()

        self._dtrajs_full = ensure_dtraj_list(dtrajs)
        self._tau = lag

        self._reversible = reversible
        # self.sliding = sliding

        # count states
        import pyemma.msm.estimation as msmest

        self._n_full = msmest.number_of_states(dtrajs)

        # sparse matrix computation wanted?
        self._sparse = sparse
        if sparse:
            self._logger.warn('Sparse mode is currently untested and might lead to errors. '
                               'I strongly suggest to use sparse=False unless you know what you are doing.')
        if self._n_full > 4000 and not sparse:
            self._logger.warn('Building a dense MSM with ' + str(self._n_full) + ' states. This can be inefficient or '
                              'unfeasible in terms of both runtime and memory consumption. Consider using sparse=True.')

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

        # run estimation unless suppressed
        self._estimated = False
        self._kwargs = kwargs
        if estimate:
            self.estimate()

        # set time step
        from pyemma.util.units import TimeUnit

        self._timeunit = TimeUnit(dt)

    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    def estimate(self):
        r"""Runs msm estimation.

        Only need to call this method if the msm was initialized with compute=False - otherwise it will have
        been called at time of initialization.

        """
        # already computed? nothing to do
        if self._estimated:
            self._logger.warn('compute is called twice. This call has no effect.')
            return

        import pyemma.msm.estimation as msmest

        # Compute count matrix
        self._C_full = msmest.count_matrix(self._dtrajs_full, self._tau, sliding=True)

        # Compute connected sets
        self._connected_sets = msmest.connected_sets(self._C_full)

        if self.connectivity == 'largest':
            # the largest connected set is the active set. This is at the same time a mapping from active to full
            self._active_set = msmest.largest_connected_set(self._C_full)
        else:
            # for 'None' and 'all' all visited states are active
            from pyemma.util.discrete_trajectories import visited_set

            self._active_set = visited_set(self._dtrajs_full)

        # back-mapping from full to lcs
        self._full2active = -1 * np.ones((self._n_full), dtype=int)
        self._full2active[self._active_set] = np.array(range(len(self._active_set)), dtype=int)

        # active set count matrix
        from pyemma.util.linalg import submatrix

        self._C_active = submatrix(self._C_full, self._active_set)
        self._nstates = self._C_active.shape[0]

        # continue sparse or dense?
        if not self._sparse:
            # converting count matrices to arrays. As a result the transition matrix and all subsequent properties
            # will be computed using dense arrays and dense matrix algebra.
            self._C_full = self._C_full.toarray()
            self._C_active = self._C_active.toarray()

        # Effective count matrix
        self._C_effective_active = self._C_active / float(self._tau)

        # Estimate transition matrix
        if self.connectivity == 'largest':
            self._T = msmest.transition_matrix(self._C_active, reversible=self._reversible, **self._kwargs)
        elif self.connectivity == 'none':
            # reversible mode only possible if active set is connected - in this case all visited states are connected
            # and thus this mode is identical to 'largest'
            if self._reversible and not msmest.is_connected(self._C_active):
                raise ValueError('Reversible MSM estimation is not possible with connectivity mode \'none\', '+
                                 'because the set of all visited states is not reversibly connected')
            self._T = msmest.transition_matrix(self._C_active, reversible=self._reversible, **self._kwargs)
        else:
            raise NotImplementedError(
                'MSM estimation with connectivity=\'self.connectivity\' is currently not implemented.')

        # compute connected dtrajs
        self._dtrajs_active = []
        for dtraj in self._dtrajs_full:
            self._dtrajs_active.append(self._full2active[dtraj])

        self._estimated = True

    ################################################################################
    # Basic attributes
    ################################################################################

    @property
    def computed(self):
        """Returns whether this msm has been estimated yet"""
        return self._estimated

    @property
    def lagtime(self):
        """
        The lag time at which the Markov model was estimated

        """
        return self._tau

    @property
    @shortcut('dtrajs_full')
    def discrete_trajectories_full(self):
        """
        A list of integer arrays with the original (unmapped) discrete trajectories:

        """
        return self._dtrajs_full

    @property
    @shortcut('dtrajs_active')
    def discrete_trajectories_active(self):
        """
        A list of integer arrays with the discrete trajectories mapped to the connectivity mode used.
        For example, for connectivity='largest', the indexes will be given within the connected set.
        Frames that are not in the connected set will be -1.

        """
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
    def effective_count_matrix(self):
        """Statistically uncorrelated transition counts within the active set of states

        You can use this count matrix for any kind of estimation, in particular it is mean to give reasonable
        error bars in uncertainty measurements (error perturbation or Gibbs sampling of the posterior).

        The effective count matrix is obtained by dividing the sliding-window count matrix by the lag time. This
        can be shown to provide a likelihood that is the geometrical average over shifted subsamples of the trajectory,
        :math:`(s_1,\:s_{tau+1},\:...),\:(s_2,\:t_{tau+2},\:...),` etc. This geometrical average converges to the
        correct likelihood in the statistical limit _[1].

        [1] Trendelkamp-Schroer B, H Wu, F Paul and F Noe. 2015:
        Reversible Markov models of molecular kinetics: Estimation and uncertainty.
        in preparation.

        """
        self._assert_estimated()
        return self._C_effective_active

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
        self._assert_estimated()
        return self._C_full

    @property
    def active_set(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        self._assert_estimated()
        return self._active_set

    @property
    def largest_connected_set(self):
        """
        The largest reversible connected set of states

        """
        self._assert_estimated()
        return self._connected_sets[0]

    @property
    def connected_sets(self):
        """
        The reversible connected sets of states, sorted by size (descending)

        """
        self._assert_estimated()
        return self._connected_sets

    ################################################################################
    # Compute derived quantities
    ################################################################################

    @property
    def active_state_fraction(self):
        """The fraction of states in the active set.

        """
        self._assert_estimated()
        return float(self._nstates) / float(self._n_full)

    @property
    def active_count_fraction(self):
        """The fraction of counts in the active set.

        """
        self._assert_estimated()
        from pyemma.util.discrete_trajectories import count_states

        hist = count_states(self._dtrajs_full)
        hist_active = hist[self._active_set]
        return float(np.sum(hist_active)) / float(np.sum(hist))

    ################################################################################
    # For general statistics
    ################################################################################

    def trajectory_weights(self):
        """Uses the MSM to assign a probability weight to each trajectory frame.

        This is a powerful function for the calculation of arbitrary observables in the trajectories one has
        started the analysis with. The stationary probability of the MSM will be used to reweigh all states.
        Returns a list of weight arrays, one for each trajectory, and with a number of elements equal to
        trajectory frames. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`, this function
        returns corresponding weights:
        .. math::
            (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})
        that are normalized to one:
        .. math::
            \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} = 1
        Suppose you are interested in computing the expectation value of a function :math:`a(x)`, where :math:`x`
        are your input configurations. Use this function to compute the weights of all input configurations and
        obtain the estimated expectation by:
        .. math::
            \langle a \rangle = \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} a(x_{i,t})
        Or if you are interested in computing the time-lagged correlation between functions :math:`a(x)` and
        :math:`b(x)` you could do:
        .. math::
            \langle a(t) b(t+\tau) \rangle_t = \sum_{i=1}^N \sum_{t=1}^{T_i} w_{i,t} a(x_{i,t}) a(x_{i,t+\tau})

        Returns
        -------
        The normalized trajectory weights. Given :math:`N` trajectories of lengths :math:`T_1` to :math:`T_N`,
        returns the corresponding weights:
        .. math::
            (w_{1,1}, ..., w_{1,T_1}), (w_{N,1}, ..., w_{N,T_N})

        """
        # compute stationary distribution, expanded to full set
        statdist_full = np.zeros([self._n_full])
        statdist_full[self._active_set] = self.stationary_distribution
        # simply read off stationary distribution and accumulate total weight
        W = []
        wtot = 0.0
        for dtraj in self._dtrajs_full:
            w = statdist_full[dtraj]
            W.append(w)
            wtot += np.sum(W)
        # normalize
        for w in W:
            w /= wtot
        # done
        return W

    ################################################################################
    # Generation of trajectories and samples
    ################################################################################

    @property
    def active_state_indexes(self):
        """
        Ensures that the connected states are indexed and returns the indices
        """
        try:  # if we have this attribute, return it
            return self._active_state_indexes
        except:  # didn't exist? then create it.
            import pyemma.util.discrete_trajectories as dt

            self._active_state_indexes = dt.index_states(self._dtrajs_full, subset=self._active_set)
            return self._active_state_indexes

    def generate_traj(self, N, start=None, stop=None, stride=1):
        """Generates a synthetic discrete trajectory of length N and simulation time stride * lag time * N

        This information can be used
        in order to generate a synthetic molecular dynamics trajectory - see
        :func:`pyemma.coordinates.save_traj`

        Note that the time different between two samples is the Markov model lag time  :math:`\tau`. When comparing
        quantities computing from this synthetic trajectory and from the input trajectories, the time points of this
        trajectory must be scaled by the lag time in order to have them on the same time scale.

        Parameters
        ----------
        N : int
            Number of time steps in the output trajectory. The total simulation time is stride * lag time * N
        start : int, optional, default = None
            starting state. If not given, will sample from the stationary distribution of P
        stop : int or int-array-like, optional, default = None
            stopping set. If given, the trajectory will be stopped before N steps
            once a state of the stop set is reached
        stride : int, optional, default = 1
            Multiple of lag time used as a time step. By default, the time step is equal to the lag time

        Returns
        -------
        indexes : ndarray( (N, 2) )
            trajectory and time indexes of the simulated trajectory. Each row consist of a tuple (i, t), where i is
            the index of the trajectory and t is the time index within the trajectory.
            Note that the time different between two samples is the Markov model lag time  :math:`\tau`.

        See also
        --------
        pyemma.coordinates.save_traj
            in order to save this synthetic trajectory as a trajectory file with molecular structures

        """
        # TODO: this is the only function left which does something time-related in a multiple of tau rather than dt.
        # TODO: we could generate dt-strided trajectories by sampling tau times from the current state, but that would
        # TODO: probably lead to a weird-looking trajectory. Maybe we could use a HMM to generate intermediate 'hidden'
        # TODO: frames. Anyway, this is a nontrivial issue.
        # generate synthetic states
        from pyemma.msm.generation import generate_traj as _generate_traj

        syntraj = _generate_traj(self._T, N, start=start, stop=stop, dt=stride)
        # result
        from pyemma.util.discrete_trajectories import sample_indexes_by_sequence

        return sample_indexes_by_sequence(self.active_state_indexes, syntraj)

    def sample_by_state(self, nsample, subset=None, replace=True):
        """Generates samples of the connected states.

        For each state in the active set of states, generates nsample samples with trajectory/time indexes.
        This information can be used in order to generate a trajectory of length nsample * nconnected using
        :func:`pyemma.coordinates.save_traj` or nconnected trajectories of length nsample each using
        :func:`pyemma.coordinates.save_traj`

        Parameters
        ----------
        N : int
            Number of time steps in the output trajectory. The total simulation time is stride * lag time * N
        nsample : int
            Number of samples per state. If replace = False, the number of returned samples per state could be smaller
            if less than nsample indexes are available for a state.
        subset : ndarray((n)), optional, default = None
            array of states to be indexed. By default all states in the connected set will be used
        replace : boolean, optional
            Whether the sample is with or without replacement
        start : int, optional, default = None
            starting state. If not given, will sample from the stationary distribution of P

        Returns
        -------
        indexes : list of ndarray( (N, 2) )
            list of trajectory/time index arrays with an array for each state.
            Within each index array, each row consist of a tuple (i, t), where i is
            the index of the trajectory and t is the time index within the trajectory.

        See also
        --------
        pyemma.coordinates.save_traj
            in order to save the sampled frames sequentially in a trajectory file with molecular structures
        pyemma.coordinates.save_trajs
            in order to save the sampled frames in nconnected trajectory files with molecular structures

        """
        # generate connected state indexes
        import pyemma.util.discrete_trajectories as dt

        return dt.sample_indexes_by_state(self.active_state_indexes, nsample, subset=subset, replace=replace)

    def sample_by_distributions(self, distributions, nsample):
        """Generates samples according to given probability distributions

        Parameters
        ----------
        distributions : list or array of ndarray ( (n) )
            m distributions over states. Each distribution must be of length n and must sum up to 1.0
        nsample : int
            Number of samples per distribution. If replace = False, the number of returned samples per state could be smaller
            if less than nsample indexes are available for a state.

        Returns
        -------
        indexes : length m list of ndarray( (nsample, 2) )
            List of the sampled indices by distribution.
            Each element is an index array with a number of rows equal to nsample, with rows consisting of a
            tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.

        """
        # generate connected state indexes
        import pyemma.util.discrete_trajectories as dt

        return dt.sample_indexes_by_distribution(self.active_state_indexes, distributions, nsample)

