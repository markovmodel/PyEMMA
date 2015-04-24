__author__ = 'noe'

import numpy as np

# EMMA imports
from msm import EstimatedMSM
from pyemma.util import statistics as stat

class SampledMSM(EstimatedMSM):

    # TODO: check all parameters, remove redundancies
    def __init__(self, dtrajs, lag, nsample=1000,
                 reversible=True, sparse=False, connectivity='largest', estimate=True, sample=True,
                 dt='1 step',
                 **kwargs):
        # superclass constructor
        EstimatedMSM.__init__(self, dtrajs, lag, reversible=reversible, sparse=sparse, connectivity=connectivity,
                              estimate=estimate, dt=dt, **kwargs)

        # set params
        self._nsample = nsample
        # set defaults
        self.confidence = 0.95

        # estimate now if requested.
        if sample:
            self.sample()

    def sample(self):
        # and additionally run the sampler
        import pyemma.msm.estimation as msmest
        self.sample_Ps,  self.sample_mus = msmest.sample_tmatrix(self.count_matrix_active, nsample=self._nsample,
                                                                 reversible=self._reversible, return_statdist=True)

    def set_confidence(self, conf):
        self.confidence = conf

    def _do_sample_eigendecomposition(self, k, ncv=None):
        """Conducts the eigenvalue decompositions for all sampled matrices.

        Stores k eigenvalues, left and right eigenvectors for all sampled matrices

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
        from pyemma.util import linalg

        # left eigenvectors
        self.sample_Ls = np.empty((self._nsample), dtype=object)
        # eigenvalues
        self.sample_eigenvalues = np.empty((self._nsample), dtype=object)
        # right eigenvectors
        self.sample_Rs = np.empty((self._nsample), dtype=object)
        # eigenvector assignments
        self.sample_eig_assignments = np.empty((self._nsample), dtype=object)

        for i in range(self.sample_Ps):
            if self._reversible:
                R, D, L = rdl_decomposition(self._T, k=k, norm='reversible', ncv=ncv)
                # everything must be real-valued
                R = R.real
                D = D.real
                L = L.real
            else:
                R, D, L = rdl_decomposition(self._T, k=k, norm='standard', ncv=ncv)
            # assign
            self.sample_Ls[i] = L
            self.sample_eigenvalues = np.diag(D)
            self.sample_Rs[i] = R
            self.sample_eig_assignments[i] = linalg.match_eigenvectors(self.eigenvectors_right(), R,
                                                                       w_ref=self.stationary_distribution,
                                                                       w=self.sample_mus[i])

    def _ensure_sample_eigendecomposition(self, k=None, ncv=None):
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
        self._assert_estimated()
        # check input?
        if self._sparse:
            if k is None:
                raise ValueError(
                    'You have requested sparse=True, then the number of eigenvalues neig must also be set.')
        else:
            # override setting - we anyway have to compute all eigenvalues, so we'll also store them.
            k = self._nstates
        # ensure that eigenvalue decomposition with k components is done.
        try:
            m = len(self._eigenvalues)  # this will raise and exception if self._eigenvalues doesn't exist yet.
            if m < k:
                # not enough eigenpairs present - recompute:
                self._do_sample_eigendecomposition(k, ncv=ncv)
        except:
            # no eigendecomposition yet - compute:
            self._do_sample_eigendecomposition(k, ncv=ncv)

    @property
    def stationary_distribution_mean(self):
        """Sample mean for the stationary distribution on the active set.

        See also
        --------
        MSM.stationary_distribution

        """
        self._assert_estimated()
        return np.mean(self.sample_mus, axis=0)

    @property
    def stationary_distribution_std(self):
        """Sample standard deviation for the stationary distribution on the active set.

        See also
        --------
        MSM.stationary_distribution

        """
        self._assert_estimated()
        return np.std(self.sample_mus, axis=0)

    @property
    def stationary_distribution_conf(self):
        """Sample confidence interval for the stationary distribution on the active set.

        See also
        --------
        MSM.stationary_distribution

        """
        self._assert_estimated()
        return stat.confidence_interval_arr(self.sample_mus, alpha=self.confidence)
