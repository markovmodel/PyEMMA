r"""Implement a MSM class that builds a Markov state models from
microstate trajectories automatically computes important properties
and provides them for later access."""

__docformat__ = "restructuredtext en"

from pyemma.msm.estimation import cmatrix, largest_connected_set, connected_cmatrix, tmatrix
from pyemma.msm.analysis import statdist, timescales

__all__ = ['MSM']


class MSM(object):

    def __init__(self, dtrajs, lag, reversible=True, sliding=True, compute=True,
                 estimate_on_lcc=True):
        r"""Estimate Markov state model (MSM) from discrete trajectories.

        Parameters
        ----------
        dtrajs : list
            discrete trajectories
        lag : int
            lagtime for the MSM estimation
        reversible : bool (optional)
            If true compute reversible MSM, else non-reversible MSM
        sliding : bool (optional)
            If true use the sliding approach to counting, else
            use the lagsampling approach
        conpute : bool (optional)
            If true estimate the MSM when creating the MSM object.
        estimate_on_lcc : bool (optional)
            If true estimate the MSM on largest connected input subset.

        Notes
        -----
        You can postpone the estimation of the MSM using compute=False and
        initiate the estimation procedure by manually calling the MSM.compute()
        method.

        """
        self.dtrajs = dtrajs
        self.lagtime = lag

        self.reversible = reversible
        self.sliding = sliding

        """Empty attributes that will be computed later"""

        """Count-matrix"""
        self.C = None

        """Largest connected set"""
        self.lcc = None

        """Count matrix on largest set"""
        self.Ccc = None

        """Transition matrix"""
        self.T = None

        """Stationary distribution"""
        self.mu = None

        self.computed = False

        self.estimate_on_lcc = estimate_on_lcc

        if compute:
            self.compute()

    def compute(self):
        """Compute count matrix"""
        self.C = cmatrix(self.dtrajs, self.lagtime, sliding=self.sliding)

        """Largest connected set"""
        self.lcc = largest_connected_set(self.C)

        """Largest connected countmatrix"""
        self.Ccc = connected_cmatrix(self.C)

        """Estimate transition matrix"""
        if self.estimate_on_lcc:
            self.T = tmatrix(self.Ccc, reversible=self.reversible)
        else:
            self.T = tmatrix(self.C, reversible=self.reversible)

        self.computed = True

    def _assert_computed(self):
        assert self.computed, "MSM hasn't been computed yet, make sure to call MSM.compute()"

    @property
    def count_matrix(self):
        self._assert_computed()
        return self.Ccc

    @property
    def count_matrix_full(self):
        self._assert_computed()
        return self.C

    @property
    def largest_connected_set(self):
        self._assert_computed()
        return self.lcc

    @property
    def transition_matrix(self):
        self._assert_computed()
        return self.T

    @property
    def stationary_distribution(self):
        self._assert_computed()
        if self.mu is None:
            self.mu = statdist(self.T)
        return self.mu

    def get_timescales(self, k):
        ts = timescales(self.T, k=k, tau=self.lagtime)
        return ts
