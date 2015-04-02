r"""Implement a MSM class that builds a Markov state models from
microstate trajectories automatically computes important properties
and provides them for later access.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

__docformat__ = "restructuredtext en"

import numpy as np

from pyemma.msm.estimation import cmatrix, largest_connected_set, connected_cmatrix, tmatrix
from pyemma.msm.analysis import statdist, timescales, eigenvectors

__all__ = ['MSM']


class MSM(object):

    def __init__(self, dtrajs, lag, reversible=True, sliding=True, compute=True,
                 estimate_on_lcc=True):
        r"""Estimate Markov state model (MSM) from discrete trajectories.

        Parameters
        ----------
        dtrajs : list containing ndarrays(dtype=int) or ndarray(n, dtype=int)
            discrete trajectories, stored as integer ndarrays (arbitrary size)
            or a single ndarray for only one trajectory.
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
        self.tau = lag

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

        """Left eigenvectors"""
        self.LEVs = None

        """Right eigenvectors"""
        self.REVs = None

        if compute:
            self.compute()

    def compute(self):
        """Compute count matrix"""
        self.C = cmatrix(self.dtrajs, self.tau, sliding=self.sliding)

        """Largest connected set"""
        self.lcc = largest_connected_set(self.C)

        """Largest connected countmatrix"""
        self.Ccc = connected_cmatrix(self.C)

        """Estimate transition matrix"""
        if self.estimate_on_lcc:
            self.T = tmatrix(self.Ccc, reversible=self.reversible)
        else:
            self.T = tmatrix(self.C, reversible=self.reversible)

        """Stationary distribution"""
        self.mu = statdist(self.T)

        """ First left and right EVs"""
        k = np.min([10, self.T.shape[0]-2])
        self.LEVs = eigenvectors(self.T, k = k, right=False )          
        self.REVs = eigenvectors(self.T, k = k, right=True )

        self.computed = True

    def _assert_computed(self):
        assert self.computed, "MSM hasn't been computed yet, make sure to call MSM.compute()"

    @property
    def discretized_trajectories(self):
        return self.dtrajs

    @property
    def lagtime(self):
        return self.tau

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
        return self.mu

    def get_timescales(self, k = 10):
        ts = timescales(self.T, k=k, tau=self.tau)
        return ts

    def get_LEVs(self, k = None):
        self._assert_computed()
        
        if k is None:
           k = self.T.shape[0] 
        else:
           k = np.min([self.T.shape[0], k])

        if self.LEVs.shape[1] < k and k < self.T.shape[0]:
           self.LEVs = eigenvectors(self.T, k, right=False )           # "some" eigenvectors: sparse diagonalization 
        elif self.LEVs.shape[1] < k and k == self.T.shape[0]:
           self.LEVs = eigenvectors(self.T.toarray(), k, right=False ) # all eigenvectors: no point in sparse diagonalization
        else:
           self.LEVs = self.LEVs[:,:k]

    def get_REVs(self, k = None):
        self._assert_computed()
        
        if k is None:
           k = self.T.shape[0] 
        else:
           k = np.min([self.T.shape[0], k])

        if self.REVs.shape[1] < k and k < self.T.shape[0]:
           self.REVs = eigenvectors(self.T, k, right=False )           # "some" eigenvectors: sparse diagonalization 
        elif self.REVs.shape[1] < k and k == self.T.shape[0]:
           self.REVs = eigenvectors(self.T.toarray(), k, right = True ) # all eigenvectors: no point in sparse diagonalization
        else:
           self.REVs = self.REVs[:,:k]
