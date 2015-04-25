__author__ = 'noe'

from pyemma.msm import estimation as msmest

class TransitionMatrixEstimator:

    def __init__(self, reversible=True, sparse=False, connectivity='largest', **kwargs):
        # set parameters
        self._reversible = reversible
        self._sparse = sparse
        self._connectivity = connectivity
        self._kwargs = kwargs
        self._estimated = False

    def estimate(self, C, return_statdist = False):
        """ Estimates transition matrix from count matrix

        Parameters
        ----------
        C : ndarray (n,n)
            active set count matrix

        Returns
        -------
        P : ndarray (n,n)
            active set transition matrix
        mu : ndarray (n)
            active set stationary distribution, only returned if return_statdist = True.

        """
        # Estimate transition matrix
        if self._connectivity == 'largest':
            self._T = msmest.transition_matrix(C, reversible=self._reversible, **self._kwargs)
        elif self._connectivity == 'none':
            # reversible mode only possible if active set is connected - in this case all visited states are connected
            # and thus this mode is identical to 'largest'
            if self._reversible and not msmest.is_connected(C):
                raise ValueError('Reversible MSM estimation is not possible with connectivity mode \'none\', '+
                                 'because the set of all visited states is not reversibly connected')
            self._T = msmest.transition_matrix(C, reversible=self._reversible, **self._kwargs)
        else:
            raise NotImplementedError(
                'MSM estimation with connectivity=\'self.connectivity\' is currently not implemented.')

        self._estimated = True
        return self._T

    def _assert_estimated(self):
        assert self._estimated, "MSM hasn't been estimated yet, make sure to call estimate()"

    @property
    def is_reversible(self):
        """Returns whether the MSM is reversible """
        return self._reversible

    @property
    def transition_matrix(self):
        """
        The transition matrix, estimated on the active set. For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        self._assert_estimated()
        return self._T

    @property
    def stationary_distribution(self):
        """The stationary distribution, estimated on the active set.

        For example, for connectivity='largest' it will be the
        transition matrix amongst the largest set of reversibly connected states

        """
        self._assert_estimated()
        try:
            return self._mu
        except:
            from pyemma.msm.analysis import stationary_distribution as _statdist

            self._mu = _statdist(self._T)
            return self._mu

