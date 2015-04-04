r"""Implement a MSM class that builds a Markov state models from
microstate trajectories automatically computes important properties
and provides them for later access.

.. moduleauthor:: F. Noe <frank DOT noe AT fu-berlin DOT de>
.. moduleauthor:: B. Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

__docformat__ = "restructuredtext en"

import numpy as np
import warnings
import copy
from pyemma.util.annotators import shortcut


__all__ = ['MSM']

# TODO - DISCUSS CHANGES:
# TODO: I removed the option sliding=True because now we force sliding but compute a transition matrix with effective
#       counts for computing error bars
# TODO: By default all operations are dense (+usability +reliability). The behavior can be switched by the user with
#       the flag sparse. a warning is generated when building an MSM > 4000 states with sparse=False
# TODO: sample trajectory and by state


# TODO: Explain concept of an active set
# TODO: Take care of sliding counts


class MSM(object):

    def __init__(self, dtrajs, lag,
                 reversible=True, sparse=False, neig=None, connectivity='largest', compute=True,
                 dt = '1 step',
                 **kwargs):
        r"""Estimate Markov state model (MSM) from discrete trajectories.

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
        neig : int, optional, default = None
            Number of eigenvalues to be computed. By default (dense) all eigenvalues will be computed. This property
            must be set if sparse is True.
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
        compute : bool, optional, default=True
            If true estimate the MSM when creating the MSM object.
        dt : str, optional, default='1 step'
            Description of the physical time corresponding to the lag. May be used by analysis algorithms such as
            plotting tools to pretty-print the axes. By default '1 step', i.e. there is no physical time unit.
            Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):
            'fs',  'femtosecond*'
            'ps',  'picosecond*'
            'ns',  'nanosecond*'
            'us',  'microsecond*'
            'ms',  'millisecond*'
            's',   'second*'

        **kwargs: Optional algorithm-specific parameters. See below for special cases
        maxiter = 1000000 : int
            Optional parameter with reversible = True.
            maximum number of iterations before the transition matrix estimation method exits
        maxerr = 1e-8 : float
            Optional parameter with reversible = True.
            convergence tolerance for transition matrix estimation.
            This specifies the maximum change of the Euclidean norm of relative
            stationary probabilities (x_i = sum_k x_ik). The relative stationary probability changes
            e_i = (x_i^(1) - x_i^(2))/(x_i^(1) + x_i^(2)) are used in order to track changes in small
            probabilities. The Euclidean norm of the change vector, |e_i|_2, is compared to convtol.

        Notes
        -----
        You can postpone the estimation of the MSM using compute=False and
        initiate the estimation procedure by manually calling the MSM.compute()
        method.

        """
        self._dtrajs_full = dtrajs
        self.tau = lag

        self.reversible = reversible
        #self.sliding = sliding

        # count states
        import pyemma.msm.estimation as msmest
        self._n_full = msmest.count_states(dtrajs)

        # sparse matrix computation wanted?
        self._sparse = sparse
        self._neig = neig
        if self._n_full > 4000 and not sparse:
            warnings.warn('Building a dense MSM with '+str(self._n_full)+' states. This can be inefficient or '+
                          'unfeasible in terms of both runtime and memory consumption. Consider using sparse=True.')
        if sparse and neig is None:
            raise ValueError('You have requested sparse=True, then the number of eigenvalues neig must also be set.')

        # store connectivity mode (lowercase)
        self.connectivity = connectivity.lower()
        if self.connectivity == 'largest':
            pass  # this is the current default. no need to do anything
        elif self.connectivity == 'all':
            raise NotImplementedError('MSM estimation with connectivity=\'all\' is currently not implemented.')
        elif self.connectivity == 'none':
            raise NotImplementedError('MSM estimation with connectivity=\'none\' is currently not implemented.')
        else:
            raise ValueError('connectivity mode '+str(connectivity)+' is unknown.')

        # run estimation unless suppressed
        self._computed = False
        if compute:
            self.compute(kwargs)

        # set time step
        from pyemma.util.units import TimeUnit
        self._timeunit = TimeUnit(dt)


    def compute(self, **kwargs):
        r"""Runs msm estimation.

        Only need to call this method if the msm was initialized with compute=False - otherwise it will have
        been called at time of initialization.

        """
        import pyemma.msm.estimation as msmest

        # Compute count matrix
        self._C_full = msmest.count_matrix(self._dtrajs_full, self.tau, sliding=True)

        # Compute connected sets
        self._connected_sets = msmest.largest_connected_set(self._C_full)

        if self.connectivity == 'largest':
            # the largest connected set is the active set. This is at the same time a mapping from active to full
            self._active_set = self._connected_sets[0]
        else:
            # for 'None' and 'all' all visited states are active
            from pyemma.util.discrete_trajectories import visited_set
            self._active_set = visited_set(self._dtrajs_full)

        # back-mapping from full to lcs
        self._full2active = -1*np.ones((self._n_full), dtype=int)
        self._full2active[self._active_set] = range(len(self._active_set))

        # active set count matrix
        from pyemma.util.linalg import submatrix
        self._C_active = submatrix(self._C_full, self._active_set)

        # continue sparse or dense?
        if not self.sparse:
            # converting count matrices to arrays. As a result the transition matrix and all subsequent properties
            # will be computed using dense arrays and dense matrix algebra.
            self._C_full = self._C_full.toarray()
            self._C_active = self._C_active.toarray()

        # Estimate transition matrix
        if self.connectivity == 'largest':
            self._T = msmest.transition_matrix(self._C_active, reversible=self.reversible, **kwargs)
        elif self.connectivity == 'none':
            # reversible mode only possible if active set is connected - in this case all visited states are connected
            # and thus this mode is identical to 'largest'
            if self.reversible and not msmest.is_connected(self._C_active):
                raise ValueError('Reversible MSM estimation is not possible with connectivity mode \'none\', because the set of all visited states is not reversibly connected')
            self._T = msmest.transition_matrix(self._C_active, reversible=self.reversible, **kwargs)
        else:
            raise NotImplementedError('MSM estimation with connectivity=\'self.connectivity\' is currently not implemented.')

        # compute connected dtrajs
        self._dtrajs_active = []
        for dtraj in self._dtrajs_full:
            self._dtrajs_active.append(self._full2active[dtraj])

        self._computed = True


    ################################################################################
    # Basic attributes
    ################################################################################

    @property
    def is_sparse(self):
        """Returns whether the MSM is sparse """
        return self._sparse

    @property
    def timestep(self):
        """Returns the time step as string, e.g. '10 ps'"""
        return str(self._timeunit)

    @property
    def computed(self):
        """Returns whether this msm has been estimated yet"""
        return self._computed

    def _assert_computed(self):
        assert self._computed, "MSM hasn't been computed yet, make sure to call MSM.compute()"

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
        return self._dtrajs_full

    @property
    def lagtime(self):
        """
        The lag time at which the Markov model was estimated

        """
        return self.tau

    @property
    def count_matrix_active(self):
        """
        The count matrix on the active set given the connectivity mode used. For example, for connectivity='largest',
        the count matrix is given only on the largest reversibly connected set.

        """
        self._assert_computed()
        return self._C_active

    @property
    def count_matrix_full(self):
        """
        The count matrix on full set of discrete states, irrespective as to whether they are connected or not.

        """
        self._assert_computed()
        return self._C_full

    @property
    def largest_connected_set(self):
        """
        The largest reversible connected set of states

        """
        self._assert_computed()
        return self._connected_sets[0]

    @property
    def connected_sets(self):
        """
        The reversible connected sets of states, sorted by size (descending)

        """
        self._assert_computed()
        return self._connected_sets

    @property
    def transition_matrix(self):
        """
        The transition matrix, estimated on the active set. For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        self._assert_computed()
        return self._T


    ################################################################################
    # Compute derived quantities
    ################################################################################

    @property
    def active_state_fraction(self):
        """The fraction of states in the active set.

        """
        self._assert_computed()
        return float(self._active_set) / float(self._n_full)

    @property
    def active_count_fraction(self):
        """The fraction of counts in the active set.

        """
        self._assert_computed()
        from pyemma.util.discrete_trajectories import count_states
        hist = count_states(self._dtrajs_full)
        hist_active = hist[self._active_set]
        return float(np.sum(hist_active)) / float(np.sum(hist))

    @property
    def stationary_distribution(self):
        """The stationary distribution, estimated on the active set.

        For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        self._assert_computed()
        try:
            return self._mu
        except:
            from pyemma.msm.analysis import stationary_distribution as _statdist
            self._mu = _statdist(self._T)
            return self._mu

    def get_timescales(self, k = None):
        """
        The relaxation timescales corresponding to the eigenvalues

        Parameters
        ----------
        k : int
            number of timescales to be computed. By default identical to the number of eigenvalues computed minus 1

        Returns
        -------
        ts : ndarray(m)
            relaxation timescales, defined by :math:`-tau / ln | \lambda_i |, i = 2,...,k+1`.

        """
        from pyemma.msm.analysis import timescales as _timescales
        ts = _timescales(self._T, k=k+1, tau=self.tau)[1:] # exclude the stationary process
        return ts

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
        statdist_full = np.zeros[self._n_full]
        statdist_full[self._active_set] = self.stationary_distribution
        # simply read off stationary distribution and accumulate total weight
        W = []
        wtot = 0.0
        for dtraj in self._dtrajs_full:
            w = statdist_full[dtraj]
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
        try: # if we have this attribute, return it
            return self._active_state_indexes
        except: # didn't exist? then create it.
            import pyemma.util.discrete_trajectories as dt
            self._active_state_indexes = dt.index_states(self._active_set)
            return self._active_state_indexes

    def generate_traj(self, N, start = None, stop = None, stride = 1):
        """Generates a synthetic discrete trajectory of length N and simulation time stride * lag time * N

        This information can be used
        in order to generate a synthetic molecular dynamics trajectory - see :py:function:`pyemma.coordinates.save_traj`

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

        See also
        --------
        :py:function:`pyemma.coordinates.save_traj`
            in order to save this synthetic trajectory as a trajectory file with molecular structures

        """
        # generate synthetic states
        from pyemma.msm.generation import generate_traj as _generate_traj
        syntraj = _generate_traj(self._T, N, start = start, stop = stop, dt = stride)
        # result
        from pyemma.util.discrete_trajectories import sample_indexes_by_sequence
        return sample_indexes_by_sequence(self.active_state_indexes, syntraj)

    def sample_by_state(self, nsample, subset=None, replace=True):
        """Generates samples of the connected states.

        For each state in the active set of states, generates nsample samples with trajectory/time indexes.
        This information can be used in order to generate a trajectory of length nsample * nconnected using
        :py:function:`pyemma.coordinates.save_traj` or nconnected trajectories of length nsample each using
        :py:function:`pyemma.coordinates.save_trajs`

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
        :py:function:`pyemma.coordinates.save_traj`
            in order to save the sampled frames sequentially in a trajectory file with molecular structures
        :py:function:`pyemma.coordinates.save_trajs`
            in order to save the sampled frames in nconnected trajectory files with molecular structures

        """
        # generate connected state indexes
        import pyemma.util.discrete_trajectories as dt
        return dt.sample_indexes_by_state(self._active_set, nsample, subset=subset, replace=replace)
