__author__ = 'noe'

import numpy as np

class SampledMSM:

    def __called_at_import_time__(self):
        self._create_attribute_mean('stationary_distribution_', 'stationary distribution on the active set',
                                    'MSM.stationary_distribution', self.sample_mus)
        self._create_attribute_std('stationary_distribution_', 'stationary distribution on the active set',
                                   'MSM.stationary_distribution', self.sample_mus)
        self._create_attribute_conf('stationary_distribution_', 'stationary distribution on the active set',
                                    'MSM.stationary_distribution', self.sample_mus)
        self._create_function_mean('eigenvectors_right_', 'right eigenvectors', 'MSM.eigenvectors_right',
                                   self._ensure_sample_eigendecomposition, [k=None, ncv=None], self.sample_Rs)
        self._create_function_std('eigenvectors_right_', 'right eigenvectors', 'MSM.eigenvectors_right',
                                  self._ensure_sample_eigendecomposition, [k=None, ncv=None], self.sample_Rs)
        self._create_function_conf('eigenvectors_right_', 'right eigenvectors', 'MSM.eigenvectors_right',
                                   self._ensure_sample_eigendecomposition, [k=None, ncv=None], self.sample_Rs)

    # ==========================
    # These are the generator functions - maybe they could be combined into two or one, and they could be located
    # elsewhere
    # ==========================

    def _create_attribute_mean(self, prefix, doctext, seealso, sample):
        # do fancy stuff here
        pass

    def _create_function_mean(self, prefix, doctext, seealso, callfirst, args, sample):
        # do fancy stuff here
        pass

    # ==========================
    # The following functions / attributes should be generated
    # ==========================

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
