__author__ = 'noe'

import numpy as np
import copy

# EMMA imports
from msm_estimated import EstimatedMSM
from pyemma.util import statistics as stat

class SampledMSM(EstimatedMSM):

    def __init__(self, estimator, sample_Ps, sample_mus, conf=0.683):
        r""" Constructs a sampled MSM

        Parameters
        ----------
        estimator : MSMEstimator
            Single-point estimator, e.g. containing a maximum likelihood or mean MSM
        sample_Ps : array-like of ndarray(n,n)
            Sampled transition matrices. They must all have the size of the active set
        sample_mus : array-like of ndarray(n)
            Sampled stationary distributions. They must all have the size of the active set
        conf : float, optional, default=0.68
            Confidence interval. By default one-sigma (68.3%) is used. Use 95.4% for two sigma or 99.7% for three sigma.

        """
        # superclass constructor
        EstimatedMSM.__init__(self, estimator)

        # set params. Make a deep copy to avoid changing from outside
        self.sample_Ps = copy.deepcopy(sample_Ps)
        self.sample_mus = copy.deepcopy(sample_mus)
        self._nsample = len(sample_Ps)
        self._confidence = conf


    def set_confidence(self, conf):
        self._confidence = conf

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

        for i in range(self._nsample):
            if self._reversible:
                R, D, L = rdl_decomposition(self.sample_Ps[i], k=k, norm='reversible', ncv=ncv)
                # everything must be real-valued
                R = R.real
                D = D.real
                L = L.real
            else:
                R, D, L = rdl_decomposition(self.sample_Ps[i], k=k, norm='standard', ncv=ncv)
            # assign ordered
            I = linalg.match_eigenvectors(self.eigenvectors_right(), R,
                                          w_ref=self.stationary_distribution, w=self.sample_mus[i])
            self.sample_Ls[i] = L[I,:]
            self.sample_eigenvalues[i] = np.diag(D)[I]
            self.sample_Rs[i] = R[:,I]

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
            m = len(self.sample_eigenvalues[0])  # this will raise and exception if self._eigenvalues doesn't exist yet.
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
        return np.mean(self.sample_mus, axis=0)

    @property
    def stationary_distribution_std(self):
        """Sample standard deviation for the stationary distribution on the active set.

        See also
        --------
        MSM.stationary_distribution

        """
        return np.std(self.sample_mus, axis=0)

    @property
    def stationary_distribution_conf(self):
        """Sample confidence interval for the stationary distribution on the active set.

        See also
        --------
        MSM.stationary_distribution

        """
        return stat.confidence_interval_arr(self.sample_mus, alpha=self._confidence)

    def eigenvalues_mean(self, k=None, ncv=None):
        """Sample mean for the eigenvalues.

        See also
        --------
        MSM.eigenvalues

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return np.mean(self.sample_eigenvalues, axis=0)

    def eigenvalues_std(self, k=None, ncv=None):
        """Sample standard deviation for the eigenvalues.

        See also
        --------
        MSM.eigenvalues

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return np.std(self.sample_eigenvalues, axis=0)

    def eigenvalues_conf(self, k=None, ncv=None):
        """Sample confidence interval for the eigenvalues.

        See also
        --------
        MSM.eigenvalues

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return stat.confidence_interval_arr(self.sample_eigenvalues, alpha=self._confidence)

    def eigenvectors_left_mean(self, k=None, ncv=None):
        """Sample mean for the left eigenvectors.

        See also
        --------
        MSM.eigenvectors_left

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return np.mean(self.sample_Ls, axis=0)

    def eigenvectors_left_std(self, k=None, ncv=None):
        """Sample standard deviation for the left eigenvectors.

        See also
        --------
        MSM.eigenvectors_left

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return np.std(self.sample_Ls, axis=0)

    def eigenvectors_left_conf(self, k=None, ncv=None):
        """Sample confidence interval for the left eigenvectors.

        See also
        --------
        MSM.eigenvectors_left

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return stat.confidence_interval_arr(self.sample_Ls, alpha=self._confidence)

    def eigenvectors_right_mean(self, k=None, ncv=None):
        """Sample mean for the right eigenvectors.

        See also
        --------
        MSM.eigenvectors_right

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return np.mean(self.sample_Rs, axis=0)

    def eigenvectors_right_std(self, k=None, ncv=None):
        """Sample standard deviation for the right eigenvectors.

        See also
        --------
        MSM.eigenvectors_right

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return np.std(self.sample_Rs, axis=0)

    def eigenvectors_right_conf(self, k=None, ncv=None):
        """Sample confidence interval for the right eigenvectors.

        See also
        --------
        MSM.eigenvectors_right

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return stat.confidence_interval_arr(self.sample_Rs, alpha=self._confidence)

    def _sample_timescales(self):
        """Compute sample timescales from the sample eigenvalues"""
        res = np.empty((self._nsample), dtype=np.object)
        for i in range(self._nsample):
            res[i] = -self._lag / np.log(np.abs(self.sample_eigenvalues[i][1:]))
        return res

    def timescales_mean(self, k=None, ncv=None):
        """Sample mean for the timescales.

        See also
        --------
        MSM.timescales

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return np.mean(self._sample_timescales(), axis=0)

    def timescales_std(self, k=None, ncv=None):
        """Sample standard deviation for the timescales.

        See also
        --------
        MSM.timescales

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return np.std(self._sample_timescales(), axis=0)

    def timescales_conf(self, k=None, ncv=None):
        """Sample confidence interval for the timescales.

        See also
        --------
        MSM.timescales

        """
        self._ensure_sample_eigendecomposition(k=k, ncv=ncv)
        return stat.confidence_interval_arr(self._sample_timescales(), alpha=self._confidence)


    def _sample_mfpt(self, A, B):
        """Compute sample timescales from the sample eigenvalues"""
        res = np.zeros((self._nsample))
        for i in range(self._nsample):
            res[i] = self._mfpt(self.sample_Ps[i], A, B, mu=self.sample_mus[i])
        return res

    def mfpt_mean(self, A, B):
        """Sample mean for the A->B mean first passage time.

        See also
        --------
        MSM.mfpt

        """
        return np.mean(self._sample_mfpt(A,B), axis=0)

    def mfpt_std(self, A, B):
        """Sample standard deviation for the A->B mean first passage time.

        See also
        --------
        MSM.mfpt

        """
        return np.std(self._sample_mfpt(A,B), axis=0)

    def mfpt_conf(self, A, B):
        """Sample confidence interval for the A->B mean first passage time.

        See also
        --------
        MSM.mfpt

        """
        return stat.confidence_interval_arr(self._sample_mfpt(A,B), alpha=self._confidence)

    def _sample_committor_forward(self, A, B):
        """Compute sample timescales from the sample eigenvalues"""
        res = np.empty((self._nsample), dtype=np.object)
        for i in range(self._nsample):
            res[i] = self._committor_forward(self.sample_Ps[i], A, B)
        return res

    def committor_forward_mean(self, A, B):
        """Sample mean for the A->B forward committor.

        See also
        --------
        MSM.committor_forward

        """
        return np.mean(self._sample_committor_forward(A,B), axis=0)

    def committor_forward_std(self, A, B):
        """Sample standard deviation for the A->B forward committor.

        See also
        --------
        MSM.committor_forward

        """
        return np.std(self._sample_committor_forward(A,B), axis=0)

    def committor_forward_conf(self, A, B):
        """Sample confidence interval for the A->B forward committor.

        See also
        --------
        MSM.committor_forward

        """
        return stat.confidence_interval_arr(self._sample_committor_forward(A,B), alpha=self._confidence)


    def _sample_committor_backward(self, A, B):
        """Compute sample timescales from the sample eigenvalues"""
        res = np.empty((self._nsample), dtype=np.object)
        for i in range(self._nsample):
            res[i] = self._committor_backward(self.sample_Ps[i], A, B, mu=self.sample_mus[i])
        return res

    def committor_backward_mean(self, A, B):
        """Sample mean for the A->B backward committor.

        See also
        --------
        MSM.committor_backward

        """
        return np.mean(self._sample_committor_backward(A,B), axis=0)

    def committor_backward_std(self, A, B):
        """Sample standard deviation for the A->B backward committor.

        See also
        --------
        MSM.committor_backward

        """
        return np.std(self._sample_committor_backward(A,B), axis=0)

    def committor_backward_conf(self, A, B):
        """Sample confidence interval for the A->B backward committor.

        See also
        --------
        MSM.committor_backward

        """
        return stat.confidence_interval_arr(self._sample_committor_backward(A,B), alpha=self._confidence)
