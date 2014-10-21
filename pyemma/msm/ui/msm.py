r"""Implement a MSM class that builds a Markov state models from
microstate trajectories automatically computes important properties
and provides them for later access."""

from pyemma.msm.estimation import cmatrix, largest_connected_set, connected_cmatrix, tmatrix
from pyemma.msm.analysis import statdist, timescales

class MSM(object):    
    def __init__(self, dtrajs, lag, reversible=True, sliding=True, compute=True):
        self.dtrajs = dtrajs
        self.lagtime = lag

        self.reversible=reversible
        self.sliding=sliding

        """Empty attributes that will be computed later"""

        """Count-matrix"""
        self.C = None

        """Largest connected set"""
        self.lcc = None

        """Count matrix on largest set"""
        self.Ccc = None

        """Tranistion matrix"""
        self.T = None

        """Stationary distribution"""
        self.mu = None

        self.computed=False

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
        self.T = tmatrix(self.Ccc, reversible=self.reversible)

        self.computed=True
    
    def _assert_computed(self):
        assert self.computed, "MSM hasn't been computed yet, make sure to call ._compute()"

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
            self.mu=statdist(self.T)
        return self.mu

    def get_timescales(self, k):
        ts=timescales(self.T, k=k, tau=self.lagtime)
        return ts
        


    

    
        






