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

from pyemma.coordinates import save_traj
from pyemma.coordinates import save_trajs

__all__ = ['MSM']

# TODO - DISCUSS CHANGES:
# TODO: I removed the option sliding=True because now we force sliding but compute a transition matrix with effective
#       counts for computing error bars
# TODO: By default all operations are dense (+usability +reliability). The behavior can be switched by the user with
#       the flag sparse. a warning is generated when building an MSM > 4000 states with sparse=False
# TODO: sample trajectory and by state


# TODO: Explain concept of an active set


class MSM(object):

    def __init__(self, dtrajs, lag,
                 reversible=True, sparse=False, connectivity='largest', compute=True,
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
            stationary probabilities (:math:`x_i = \sum_k x_{ik}`). The relative stationary probability changes
            :math:`e_i = (x_i^{(1)} - x_i^{(2)})/(x_i^{(1)} + x_i^{(2)})` are used in order to track changes in small
            probabilities. The Euclidean norm of the change vector, :math:`|e_i|_2`, is compared to maxerr.

        Notes
        -----
        You can postpone the estimation of the MSM using compute=False and
        initiate the estimation procedure by manually calling the MSM.compute()
        method.

        """
        # TODO: extensive input checking!
        from pyemma.util.types import ensure_dtraj_list
        self._dtrajs_full = ensure_dtraj_list(dtrajs)
        self._tau = lag

        self._reversible = reversible
        #self.sliding = sliding

        # count states
        import pyemma.msm.estimation as msmest
        self._n_full = msmest.number_of_states(dtrajs)

        # sparse matrix computation wanted?
        self._sparse = sparse
        if self._n_full > 4000 and not sparse:
            warnings.warn('Building a dense MSM with '+str(self._n_full)+' states. This can be inefficient or '+
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
            raise ValueError('connectivity mode '+str(connectivity)+' is unknown.')

        # run estimation unless suppressed
        self._computed = False
        self._kwargs = kwargs
        if compute:
            self.compute()

        # set time step
        from pyemma.util.units import TimeUnit
        self._timeunit = TimeUnit(dt)



    def compute(self):
        r"""Runs msm estimation.

        Only need to call this method if the msm was initialized with compute=False - otherwise it will have
        been called at time of initialization.

        """
        # already computed? nothing to do
        if self._computed:
            warnings.warn('compute is called twice. This call has no effect.')
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
        self._full2active = -1*np.ones((self._n_full), dtype=int)
        self._full2active[self._active_set] = np.array(range(len(self._active_set)), dtype=int)

        # active set count matrix
        from pyemma.util.linalg import submatrix
        self._C_active = submatrix(self._C_full, self._active_set)
        self._n_active = self._C_active.shape[0]

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
                raise ValueError('Reversible MSM estimation is not possible with connectivity mode \'none\', because the set of all visited states is not reversibly connected')
            self._T = msmest.transition_matrix(self._C_active, reversible=self._reversible, **self._kwargs)
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
    def is_reversible(self):
        """Returns whether the MSM is sparse """
        return self._reversible

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
        return self._dtrajs_active

    @property
    def lagtime(self):
        """
        The lag time at which the Markov model was estimated

        """
        return self._tau

    @property
    def count_matrix_active(self):
        """The count matrix on the active set given the connectivity mode used.

        For example, for connectivity='largest', the count matrix is given only on the largest reversibly connected set.
        Attention: This count matrix has been obtained by sliding a window of length tau across the data. It contains
        a factor of tau more counts than are statistically uncorrelated. It's fine to use this matrix for maximum
        likelihood estimated, but it will give far too small errors if you use it for uncertainty calculations. In order
        to do uncertainty calculations, use the effective count matrix, see:
        effective_count_matrix

        See Also
        --------
        effective_count_matrix
            For a count matrix with effective (statistically uncorrelated) counts.

        """
        self._assert_computed()
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
        self._assert_computed()
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
        self._assert_computed()
        return self._C_full

    @property
    def nstates(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        self._assert_computed()
        return self._n_active

    @property
    def active_set(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        self._assert_computed()
        return self._active_set

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
        return float(self._n_active) / float(self._n_full)

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


    def _do_eigendecomposition(self, k, ncv=None):
        """Conducts the eigenvalue decomposition and stores k eigenvalues, left and right eigenvectors

        Parameters
        ----------
        k : int
            The number of eigenvalues / eigenvectors to be kept
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        """
        from pyemma.msm.analysis import rdl_decomposition
        if self._reversible:
            self._R, self._D, self._L = rdl_decomposition(self._T, k=k, norm='reversible', ncv=ncv)
            # everything must be real-valued
            self._R = self._R.real
            self._D = self._D.real
            self._L = self._L.real
        else:
            self._R, self._D, self._L = rdl_decomposition(self._T, k=k, norm='normal', ncv=ncv)
        self._eigenvalues = np.diag(self._D)


    def _ensure_eigendecomposition(self, k = None, ncv=None):
        """Ensures that eigendecomposition has been performed with at least k eigenpairs

        k : int
            number of eigenpairs needed. This setting is mandatory for sparse transition matrices
            (if you set sparse=True in the initialization). For dense matrices, k will be ignored
            as all eigenvalues and eigenvectors will be computed and stored.
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        """
        # are we ready?
        self._assert_computed()
        # check input?
        if self._sparse:
            if k is None:
                raise ValueError('You have requested sparse=True, then the number of eigenvalues neig must also be set.')
        else:
            # override setting - we anyway have to compute all eigenvalues, so we'll also store them.
            k = self._n_active
        # ensure that eigenvalue decomposition with k components is done.
        try:
            m = len(self._eigenvalues) # this will raise and exception if self._eigenvalues doesn't exist yet.
            if (m < k):
                # not enough eigenpairs present - recompute:
                self._do_eigendecomposition(k, ncv=ncv)
        except:
            # no eigendecomposition yet - compute:
            self._do_eigendecomposition(k, ncv=ncv)


    def eigenvalues(self, k = None, ncv = None):
        """Compute the transition matrix eigenvalues

        Parameters
        ----------
        k : int
            number of timescales to be computed. By default identical to the number of eigenvalues computed minus 1
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        ts : ndarray(m)
            transition matrix eigenvalues :math:`\lambda_i, i = 1,...,k`., sorted by descending norm.

        """
        self._ensure_eigendecomposition(k = k, ncv = ncv)
        return self._eigenvalues[:k]


    def eigenvectors_left(self, k = None, ncv = None):
        """Compute the left transition matrix eigenvectors

        Parameters
        ----------
        k : int
            number of timescales to be computed. By default identical to the number of eigenvalues computed minus 1
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        L : ndarray(k,n)
            left eigenvectors in a row matrix. l_ij is the j'th component of the i'th left eigenvector

        """
        self._ensure_eigendecomposition(k = k, ncv = ncv)
        return self._L[:k,:]


    def eigenvectors_right(self, k = None, ncv = None):
        """Compute the right transition matrix eigenvectors

        Parameters
        ----------
        k : int
            number of timescales to be computed. By default identical to the number of eigenvalues computed minus 1
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        R : ndarray(n,k)
            right eigenvectors in a column matrix. r_ij is the i'th component of the j'th right eigenvector

        """
        self._ensure_eigendecomposition(k = k, ncv = ncv)
        return self._R[:,:k]


    def timescales(self, k = None, ncv = None):
        """
        The relaxation timescales corresponding to the eigenvalues

        Parameters
        ----------
        k : int
            number of timescales to be computed. As a result, k+1 eigenvalues will be computed
        ncv : int (optional)
            Relevant for eigenvalue decomposition of reversible transition matrices.
            ncv is the number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        ts : ndarray(m)
            relaxation timescales, defined by :math:`-tau / ln | \lambda_i |, i = 2,...,k+1`.

        """
        neig = k
        if (k is not None):
            neig += 1
        self._ensure_eigendecomposition(k = neig, ncv = ncv)
        from pyemma.msm.analysis.dense.decomposition import timescales_from_eigenvalues as _timescales
        ts = _timescales(self._eigenvalues, tau=self._tau)
        if neig is None:
            return ts[1:]
        else:
            return ts[1:neig] # exclude the stationary process


    def _assert_in_active(self, A):
        """
        Checks if set A is within the active set

        Parameters
        ----------
        A : int or int array
            set of states
        """
        assert np.max(A) < self._n_active, 'Chosen set contains states that are not included in the active set.'

    def mfpt(self, A, B):
        """Mean first passage times from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        self._assert_computed()
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import mfpt as _mfpt
        # scale mfpt by lag time
        return self._tau * _mfpt(self._T, B, origin=A, mu=self.stationary_distribution)


    def committor_forward(self, A, B):
        """Forward committor (also known as p_fold or splitting probability) from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        self._assert_computed()
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import committor as _committor
        return _committor(self._T, A, B, forward=True)


    def committor_backward(self, A, B):
        """Backward committor from set A to set B

        Parameters
        ----------
        A : int or int array
            set of starting states
        B : int or int array
            set of target states
        """
        self._assert_computed()
        self._assert_in_active(A)
        self._assert_in_active(B)
        from pyemma.msm.analysis import committor as _committor
        return _committor(self._T, A, B, forward=False, mu=self.stationary_distribution)


    def expectation(self, a):
        r"""Equilibrium expectation value of a given observable.

        Parameters
        ----------
        a : (M,) ndarray
            Observable vector

        Returns
        -------
        val: float
            Equilibrium expectation value fo the given observable

        Notes
        -----
        The equilibrium expectation value of an observable a is defined as follows

        .. math::

            \mathbb{E}_{\mu}[a] = \sum_i \mu_i a_i

        :math:`\mu=(\mu_i)` is the stationary vector of the transition matrix :math:`T`.
        """
        # are we ready?
        self._assert_computed()
        # check input
        assert np.shape(a)[0] == self._n_active, 'observable vector a does not have the same size like the active set. Need len(a) = '+str(self._n_active)
        return np.dot(a, self.stationary_distribution)


    def correlation(self, a, times, b=None, k=None, ncv=None):
        r"""Time-correlation for equilibrium experiment.

        Parameters
        ----------
        a : (M,) ndarray
            Observable, represented as vector on state space
        times : int or int array
            List of lag time or lag times (in units of the transition matrix lag time tau) at which to compute correlation
        b : (M,) ndarray (optional)
            Second observable, for cross-correlations
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition. The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        correlations : ndarray
            Correlation values at given times

        """
        # are we ready?
        self._assert_computed()
        # check input
        assert np.shape(a)[0] == self._n_active, 'observable vector a does not have the same size like the active set. Need len(a) = '+str(self._n_active)
        if b is not None:
            assert np.shape(b)[0] == self._n_active, 'observable vector b does not have the same size like the active set. Need len(b) = '+str(self._n_active)
        if isinstance(times, int):
            times = [times]
        from pyemma.msm.analysis import correlation as _correlation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        return _correlation(self._T, a, obs2=b, times=times, k = k, ncv = ncv)


    def fingerprint_correlation(self, a, b=None, k=None, ncv=None):
        r"""Dynamical fingerprint for equilibrium time-correlation experiment.

        Parameters
        ----------
        a : (M,) ndarray
            Observable, represented as vector on state space
        b : (M,) ndarray (optional)
            Second observable, for cross-correlations
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition. The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        timescales : (N,) ndarray
            Time-scales of the transition matrix
        amplitudes : (N,) ndarray
            Amplitudes for the correlation experiment

        References
        ----------
        .. [1] Noe, F, S Doose, I Daidone, M Loellmann, M Sauer, J D
            Chodera and J Smith. 2010. Dynamical fingerprints for probing
            individual relaxation processes in biomolecular dynamics with
            simulations and kinetic experiments. PNAS 108 (12): 4822-4827.

        """
        # are we ready?
        self._assert_computed()
        # check input
        assert np.shape(a)[0] == self._n_active, 'observable vector a does not have the same size like the active set. Need len(a) = '+str(self._n_active)
        if b is not None:
            assert np.shape(b)[0] == self._n_active, 'observable vector b does not have the same size like the active set. Need len(b) = '+str(self._n_active)
        from pyemma.msm.analysis import fingerprint_correlation as _fc
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        return _fc(self._T, a, obs2=b, tau = self._tau, k = k, ncv = ncv)


    def relaxation(self, p0, a, times, k=None, ncv=None):
        r"""Relaxation experiment.

        The relaxation experiment describes the time-evolution
        of an expectation value starting in a non-equilibrium
        situation.

        Parameters
        ----------
        p0 : (n,) ndarray
            Initial distribution for a relaxation experiment
        a : (n,) ndarray
            Observable, represented as vector on state space
        times : int or list of int (optional)
            Time steps (multiples of the transition matrix lag time tau) at which to compute expectation
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition.
            The number of Lanczos vectors generated, `ncv` must be greater than k; it is recommended that ncv > 2*k

        Returns
        -------
        res : ndarray
            Array of expectation value at given times

        """
        # are we ready?
        self._assert_computed()
        # check input
        assert np.shape(p0)[0] == self._n_active, 'initial distribution p0 does not have the same size like the active set. Need len(p0) = '+str(self._n_active)
        assert np.shape(a)[0] == self._n_active, 'observable vector a does not have the same size like the active set. Need len(a) = '+str(self._n_active)
        if isinstance(times, int):
            times = [times]
        from pyemma.msm.analysis import relaxation as _relaxation
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        return _relaxation(self._T, p0, a, times=times, k = k, ncv = ncv)


    def fingerprint_relaxation(self, p0, a, k=None, ncv=None):
        r"""Dynamical fingerprint for perturbation/relaxation experiment.

        Parameters
        ----------
        p0 : (n,) ndarray
            Initial distribution for a relaxation experiment
        a : (n,) ndarray
            Observable, represented as vector on state space
        lag : int or int array
            List of lag time or lag times (in units of the transition matrix lag time tau) at which to compute correlation
        k : int (optional)
            Number of eigenvalues and eigenvectors to use for computation
        ncv : int (optional)
            Only relevant for sparse matrices and large lag times, where the relaxation will be computes using an
            eigenvalue decomposition. The number of Lanczos vectors generated, `ncv` must be greater than k;
            it is recommended that ncv > 2*k

        Returns
        -------
        timescales : (N,) ndarray
            Time-scales of the transition matrix
        amplitudes : (N,) ndarray
            Amplitudes for the relaxation experiment

        References
        ----------
        .. [1] Noe, F, S Doose, I Daidone, M Loellmann, M Sauer, J D
            Chodera and J Smith. 2010. Dynamical fingerprints for probing
            individual relaxation processes in biomolecular dynamics with
            simulations and kinetic experiments. PNAS 108 (12): 4822-4827.

        """
        # are we ready?
        self._assert_computed()
        # check input
        assert np.shape(p0)[0] == self._n_active, 'initial distribution p0 does not have the same size like the active set. Need len(p0) = '+str(self._n_active)
        assert np.shape(a)[0] == self._n_active, 'observable vector a does not have the same size like the active set. Need len(a) = '+str(self._n_active)
        from pyemma.msm.analysis import fingerprint_relaxation as _fr
        # TODO: this could be improved. If we have already done an eigenvalue decomposition, we could provide it.
        # TODO: for this, the correlation function must accept already-available eigenvalue decompositions.
        return _fr(self._T, p0, a, tau = self._tau, k = k, ncv = ncv)


    ################################################################################
    # pcca
    ################################################################################

    def _ensure_pcca(self, m):
        # can we do it?
        if not self._reversible:
            raise ValueError('Cannot compute PCCA for non-reversible matrices. Set reversible=True when constructing the MSM.')

        from pyemma.msm.analysis.dense.pcca import PCCA
        # ensure that we have a pcca object with the right number of states
        try:
            # this will except if we don't have a pcca object
            if self.pcca.n_metastable != m:
                # incorrect number of states - recompute
                self.pcca = PCCA(self._T, m)
        except:
            # didn't have a pcca object yet - compute
            self.pcca = PCCA(self._T, m)

    def pcca_memberships(self, m):
        """ Computes the memberships of active set states to metastable sets with the PCCA++ method _[1].

        Parameters
        ----------
        m : int
            Number of Perron-clusters

        Returns
        -------
        M : ndarray((n,m))
            A matrix containing the probability or membership of each state to be assigned to each metastable set.
            i.e. p(metastable | state). The row sums of M are 1.

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179
        """
        # are we ready?
        self._assert_computed()
        self._ensure_pcca(m)
        return self.pcca.memberships

    def pcca_distributions(self, m):
        """ Computes the probability distributions of active set states within each metastable set using the PCCA++ method _[1]
        using Bayesian inversion as described in _[2].

        Parameters
        ----------
        m : int
            Number of Perron-clusters

        Returns
        -------
        p_out : ndarray((m,n))
            A matrix containing the probability distribution of each active set state, given that we are in a
            metastable set.
            i.e. p(state | metastable). The row sums of p_out are 1.

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179
        .. [2] F. Noe, H. Wu, J.-H. Prinz and N. Plattner:
            Projected and hidden Markov models for calculating kinetics and metastable states of complex molecules
            J. Chem. Phys. 139, 184114 (2013)
        """
        # are we ready?
        self._assert_computed()
        self._ensure_pcca(m)
        return self.pcca.output_probabilities

    def pcca_sets(self, m):
        """ Computes the metastable sets of active set states within each metastable set using the PCCA++ method _[1]

        This is only recommended for visualization purposes. You *cannot* compute any
        actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

        Parameters
        ----------
        m : int
            Number of Perron-clusters

        Returns
        -------
        A list of length equal to metastable states. Each element is an array with microstate indexes contained in it

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179
        """
        # are we ready?
        self._assert_computed()
        self._ensure_pcca(m)
        return self.pcca.metastable_sets

    def pcca_assignments(self, m):
        """ Computes the assignment to metastable sets for active set states using the PCCA++ method _[1]

        This is only recommended for visualization purposes. You *cannot* compute any
        actual quantity of the coarse-grained kinetics without employing the fuzzy memberships!

        Parameters
        ----------
        m : int
            Number of Perron-clusters

        Returns
        -------
        For each active set state, the metastable state it is located in.

        References
        ----------
        .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
            PCCA+: application to Markov state models and data
            classification. Advances in Data Analysis and Classification 7
            (2): 147-179
        """
        # are we ready?
        self._assert_computed()
        self._ensure_pcca(m)
        return self.pcca.metastable_assignment


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
        try: # if we have this attribute, return it
            return self._active_state_indexes
        except: # didn't exist? then create it.
            import pyemma.util.discrete_trajectories as dt
            self._active_state_indexes = dt.index_states(self._dtrajs_full, subset=self._active_set)
            return self._active_state_indexes

    def generate_traj(self, N, start = None, stop = None, stride = 1):
        """Generates a synthetic discrete trajectory of length N and simulation time stride * lag time * N

        This information can be used
        in order to generate a synthetic molecular dynamics trajectory - see
        :func:`pyemma.coordinates.save_traj`

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
        pyemma.coordinates.save_traj
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
        :meth:`pyemma.coordinates.api.save_traj`
            in order to save the sampled frames sequentially in a trajectory file with molecular structures
        :meth:`pyemma.coordinates.save_trajs`
            in order to save the sampled frames in nconnected trajectory files with molecular structures

        """
        # generate connected state indexes
        import pyemma.util.discrete_trajectories as dt
        return dt.sample_indexes_by_state(self.active_state_indexes, nsample, subset=subset, replace=replace)
