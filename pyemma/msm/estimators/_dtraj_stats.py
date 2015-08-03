__author__ = 'noe'

import numpy as np

from pyemma.msm import estimation as msmest
from pyemma.util.annotators import shortcut
from pyemma.util.linalg import submatrix
from pyemma.util.discrete_trajectories import visited_set


class DiscreteTrajectoryStats(object):
    r""" Statistics, count matrices and connectivity from discrete trajectories

    Operates sparse by default.

    """

    def __init__(self, dtrajs):
        # TODO: extensive input checking!
        from pyemma.util.types import ensure_dtraj_list

        # discrete trajectories
        self._dtrajs = ensure_dtraj_list(dtrajs)

        # basic count statistics
        import pyemma.msm.estimation as msmest
        # histogram
        self._hist = msmest.count_states(self._dtrajs)
        # total counts
        self._total_count = np.sum(self._hist)
        # number of states
        self._nstates = msmest.number_of_states(dtrajs)

        # not yet estimated
        self._counted_at_lag = False


    def count_lagged(self, lag, count_mode='sliding'):
        r""" Counts transitions at given lag time

        Parameters
        ----------
        count_mode : str, optional, default='sliding'
            mode to obtain count matrices from discrete trajectories. Should be one of:

            * 'sliding' : A trajectory of length T will have :math:`T-tau` counts
                at time indexes
                .. math:
                    (0 \rightarray \tau), (1 \rightarray \tau+1), ..., (T-\tau-1 \rightarray T-1)

            * 'effective' : Uses an estimate of the transition counts that are
                statistically uncorrelated. Recommended when used with a
                Bayesian MSM.

            * 'sample' : A trajectory of length T will have :math:`T/tau` counts
                at time indexes
                .. math:
                    (0 \rightarray \tau), (\tau \rightarray 2 \tau), ..., (((T/tau)-1) \tau \rightarray T)


        """
        # store lag time
        self._lag = lag

        # Compute count matrix
        count_mode = count_mode.lower()
        if count_mode == 'sliding':
            self._C = msmest.count_matrix(self._dtrajs, lag, sliding=True)
        elif count_mode == 'sample':
            self._C = msmest.count_matrix(self._dtrajs, lag, sliding=False)
        elif count_mode == 'effective':
            self._C = msmest.effective_count_matrix(self._dtrajs, lag)
        else:
            raise ValueError('Count mode ' + count_mode + ' is unknown.')

        # Compute reversibly connected sets
        self._connected_sets = msmest.connected_sets(self._C)

        # set sizes and count matrices on reversibly connected sets
        self._connected_set_sizes = np.zeros((len(self._connected_sets)))
        self._C_sub = np.empty((len(self._connected_sets)), dtype=np.object)
        for i in range(len(self._connected_sets)):
            # set size
            self._connected_set_sizes[i] = len(self._connected_sets[i])
            # submatrix
            self._C_sub[i] = submatrix(self._C, self._connected_sets[i])

        # largest connected set
        lcs = self._connected_sets[0]

        # mapping from full to lcs
        self._full2lcs = -1 * np.ones((self._nstates), dtype=int)
        self._full2lcs[lcs] = np.array(range(len(lcs)), dtype=int)

        # remember that this function was called
        self._counted_at_lag = True

    # ==================================
    # Permanent properties
    # ==================================

    def _assert_counted_at_lag(self):
        assert self._counted_at_lag, \
            "You haven't run count_lagged yet. Do that first before accessing lag-based quantities"

    def _assert_subset(self, A):
        """
        Checks if set A is a subset of states

        Parameters
        ----------
        A : int or int array
            set of states
        """
        assert np.max(A) < self._nstates, 'Chosen set contains states that are not included in the data.'

    @property
    def nstates(self):
        return self._nstates

    @property
    @shortcut('dtrajs')
    def discrete_trajectories(self):
        """
        A list of integer arrays with the original (unmapped) discrete trajectories:

        """
        return self._dtrajs

    @property
    def total_count(self):
        return self._hist.sum()

    @property
    @shortcut('hist')
    def histogram(self):
        r""" Histogram of discrete state counts

        """
        return self._hist

    # ==================================
    # Estimated properties
    # ==================================

    @property
    def lag(self):
        """
        The active set of states on which all computations and estimations will be done

        """
        self._assert_counted_at_lag()
        return self._lag

    def count_matrix(self, connected_set=None, subset=None, effective=False):
        """The count matrix

        Parameters
        ----------
        connected_set : int or None, optional, default=None
            connected set index. See :func:`connected_sets` to get a sorted list of connected sets.
            This parameter is exclusive with subset.
        subset : array-like of int or None, optional, default=None
            subset of states to compute the count matrix on. This parameter is exclusive with subset.
        effective : bool, optional, default=False
            Statistically uncorrelated transition counts within the active set of states

            You can use this count matrix for any kind of estimation, in particular it is mean to give reasonable
            error bars in uncertainty measurements (error perturbation or Gibbs sampling of the posterior).

            The effective count matrix is obtained by dividing the sliding-window count matrix by the lag time. This
            can be shown to provide a likelihood that is the geometrical average over shifted subsamples of the trajectory,
            :math:`(s_1,\:s_{tau+1},\:...),\:(s_2,\:t_{tau+2},\:...),` etc. This geometrical average converges to the
            correct likelihood in the statistical limit _[1].

        References
        ----------

        ..[1] Trendelkamp-Schroer B, H Wu, F Paul and F Noe. 2015:
            Reversible Markov models of molecular kinetics: Estimation and uncertainty.
            in preparation.
        """
        self._assert_counted_at_lag()
        if subset is not None and connected_set is not None:
            raise ValueError('Can\'t set both connected_set and subset.')
        if subset is not None:
            self._assert_subset(subset)
            C = submatrix(self._C, subset)
        elif connected_set is not None:
            C = self._C_sub[connected_set]
        else: # full matrix wanted
            C = self._C

        # effective count matrix wanted?
        if effective:
            C /= float(self._lag)

        return C

    @shortcut('hist_lagged')
    def histogram_lagged(self, connected_set=None, subset=None, effective=False):
        r""" Histogram of discrete state counts

        """
        C = self.count_matrix(connected_set=connected_set, subset=subset, effective=effective)
        return C.sum(axis=1)

    @property
    def total_count_lagged(self, connected_set=None, subset=None, effective=False):
        h = self.histogram_lagged(connected_set=connected_set, subset=subset, effective=effective)
        return h.sum()

    @property
    def count_matrix_largest(self, effective=False):
        """The count matrix on the largest connected set

        """
        return self.count_matrix(connected_set=0, effective=effective)

    @property
    def largest_connected_set(self):
        """
        The largest reversible connected set of states

        """
        self._assert_counted_at_lag()
        return self._connected_sets[0]

    @property
    def visited_set(self):
        r""" The set of visited states
        """
        return visited_set(self._dtrajs)

    @property
    def connected_sets(self):
        """
        The reversible connected sets of states, sorted by size (descending)

        """
        self._assert_counted_at_lag()
        return self._connected_sets

    @property
    def connected_set_sizes(self):
        """The numbers of state for each connected set

        """
        self._assert_counted_at_lag()
        return self._connected_set_sizes

