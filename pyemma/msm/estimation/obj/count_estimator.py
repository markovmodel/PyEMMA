__author__ = 'noe'

import numpy as np

from pyemma.msm import estimation as msmest
from pyemma.util.annotators import shortcut
from pyemma.util.log import getLogger
from pyemma.util.linalg import submatrix
from pyemma.util.discrete_trajectories import visited_set

class CountEstimator:
    """
    Maximum-likelihood Markov model estimator

    Estimates the transition matrix and several basic quantities from discrete trajectories.
    Supports reversible or non-reversible, sparse or dense and several connectivity modes

    Parameters
    ----------
    lag = None, int
        lag time. If given will run the estimate immediately, otherwise will just setup the estimator and hold
        until estimate(lag) is called

    """

    def __init__(self, dtrajs, lag=None, sparse=False, connectivity='largest'):
        # TODO: extensive input checking!
        from pyemma.util.types import ensure_dtraj_list

        # start logging
        self.__create_logger()

        # discrete trajectories
        self._dtrajs_full = ensure_dtraj_list(dtrajs)

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

        # run count matrix estimation if wanted
        if lag is None:
            self._estimated = False
        else:
            self.estimate(lag)

    def __create_logger(self):
        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    def estimate(self, lag):
        r"""Runs msm estimation with given lag time.

        Only need to call this method if the msm was initialized with compute=False - otherwise it will have
        been called at time of initialization.

        """
        self._lag = lag

        # Compute count matrix
        self._C_full = msmest.count_matrix(self._dtrajs_full, lag, sliding=True)

        # Compute connected sets
        self._connected_sets = msmest.connected_sets(self._C_full)

        if self.connectivity == 'largest':
            # the largest connected set is the active set. This is at the same time a mapping from active to full
            self._active_set = msmest.largest_connected_set(self._C_full)
        else:
            # for 'None' and 'all' all visited states are active
            self._active_set = visited_set(self._dtrajs_full)

        # back-mapping from full to lcs
        self._full2active = -1 * np.ones((self._n_full), dtype=int)
        self._full2active[self._active_set] = np.array(range(len(self._active_set)), dtype=int)

        # active set count matrix
        self._C_active = submatrix(self._C_full, self._active_set)
        self._nstates = self._C_active.shape[0]

        # continue sparse or dense?
        if not self._sparse:
            # converting count matrices to arrays. As a result the transition matrix and all subsequent properties
            # will be computed using dense arrays and dense matrix algebra.
            self._C_full = self._C_full.toarray()
            self._C_active = self._C_active.toarray()

        # Effective count matrix
        self._C_effective_active = self._C_active / float(lag)

        self._estimated = True

    # ==================================
    # Permanent properties
    # ==================================

    def _assert_estimated(self):
        assert self._estimated, "MSM hasn't been estimated yet, make sure to call estimate()"

    @property
    def estimated(self):
        """Returns whether this msm has been estimated yet"""
        return self._estimated

    @property
    def is_sparse(self):
        """Returns whether the MSM is sparse """
        return self._sparse

    @property
    def nstates_full(self):
        return self._n_full

    @property
    @shortcut('dtrajs_full')
    def discrete_trajectories_full(self):
        """
        A list of integer arrays with the original (unmapped) discrete trajectories:

        """
        return self._dtrajs_full

    # ==================================
    # Estimated properties
    # ==================================

    @property
    def lag(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        self._assert_estimated()
        return self._lag

    @property
    def nstates(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        self._assert_estimated()
        return self._nstates

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
        for dtraj in self._dtrajs_full:
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