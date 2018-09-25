from abc import abstractproperty

import numpy as np
from pyemma.util.annotators import deprecated

from pyemma._base.model import Model
from pyemma._base.serialization.serialization import SerializableMixIn

from pyemma.coordinates.data._base.transformer import StreamingEstimationTransformer

__author__ = 'marscher'


class TICAModelBase(Model, SerializableMixIn):
    __serialize_version = 1
    # TODO: provide patch for versino 0!

    _DEFAULT_VARIANCE_CUTOFF = 0.95

    def __init__(self, mean=None, cov=None, cov_tau=None, dim=None, epsilon=1e-6, scaling=None, lag=0):
        self.set_model_params(mean=mean, cov=cov, cov_tau=cov_tau, dim=dim, epsilon=epsilon, scaling=scaling, lag=lag)

    def set_model_params(self, mean=None, cov_tau=None, cov=None,
                         # deprecated since 2.5.5
                         cumvar=None, eigenvalues=None, eigenvectors=None,
                         # new since version 2.5.5
                         dim=_DEFAULT_VARIANCE_CUTOFF,
                         epsilon=1e-6,
                         scaling='kinetic_map',
                         lag=0,
                         ):
        self.cov = cov
        self.cov_tau = cov_tau
        self.mean = mean
        if cumvar is not None:
            raise
        if eigenvalues is not None:
            raise
        if eigenvectors is not None:
            raise
        # new since 2.5.5
        self.dim = dim
        self.epsilon = epsilon
        self.lag = lag
        self.scaling = scaling

        self._diagonalized = False

    @property
    def scaling(self):
        return self._scaling

    @scaling.setter
    def scaling(self, value):
        valid = ('kinetic_map', 'commute_map', None)
        if value not in valid:
            raise ValueError('Valid settings for scaling are one of {valid}, but was {invalid}'
                             .format(valid=valid, invalid=value))
        self._scaling = value

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, value):
        self._diagonalized = False
        self._cov = value

    @property
    def cov_tau(self):
        return self._cov_tau

    @cov_tau.setter
    def cov_tau(self, value):
        self._diagonalized = False
        self._cov_tau = value

    @property
    def eigenvectors(self):
        if not self._diagonalized:
            self._diagonalize()
        return self._eigenvectors

    @property
    def eigenvalues(self):
        if not self._diagonalized:
            self._diagonalize()
        return self._eigenvalues

    @staticmethod
    def _cumvar(eigenvalues):
        cumvar = np.cumsum(eigenvalues ** 2)
        cumvar /= cumvar[-1]
        return cumvar

    @property
    def cumvar(self):
        """ cumulative kinetic variance """
        return TICAModelBase._cumvar(self.eigenvalues)

    @staticmethod
    def _dimension(rank, dim, eigenvalues):
        """ output dimension """
        if dim is None or (isinstance(dim, float) and dim == 1.0):
            return rank
        if isinstance(dim, float):
            # subspace_variance, reduce the output dimension if needed
            return min(len(eigenvalues), np.searchsorted(TICAModelBase._cumvar(eigenvalues), dim) + 1)
        else:
            return np.min([rank, dim])

    def dimension(self):
        """ output dimension """
        if self.cov is None:  # no data yet
            if isinstance(self.dim, int):  # return user choice
                import warnings
                warnings.warn('Returning user-input for dimension, since this model has not yet been estimated.')
                return self.dim
            raise RuntimeError('Please call set_model_params prior using this method.')

        if not self._diagonalized:
            self._diagonalize()
        return self._dimension(self._rank, self.dim, self.eigenvalues)

    def _compute_diag(self):
        from pyemma._ext.variational.util import ZeroRankError
        from pyemma._ext.variational import eig_corr
        try:
            eigenvalues, eigenvectors, rank = eig_corr(self.cov, self.cov_tau, self.epsilon,
                                                             sign_maxelement=True, return_rank=True)
        except ZeroRankError:
            raise ZeroRankError('All input features are constant in all time steps. '
                                'No dimension would be left after dimension reduction.')
        return eigenvalues, eigenvectors, rank

    def _diagonalize(self):
        # diagonalize with low rank approximation
        eigenvalues, eigenvectors, self._rank = self._compute_diag()
        if self.scaling == 'kinetic_map':  # scale by eigenvalues
            eigenvectors *= eigenvalues[None, :]
        elif self.scaling == 'commute_map':  # scale by (regularized) timescales
            timescales = 1-self.lag / np.log(np.abs(eigenvalues))
            # dampen timescales smaller than the lag time, as in section 2.5 of ref. [5]
            regularized_timescales = 0.5 * timescales * np.maximum(np.tanh(np.pi * ((timescales - self.lag) / self.lag) + 1), 0)

            eigenvectors *= np.sqrt(regularized_timescales / 2)

        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._diagonalized = True


class TICABase(StreamingEstimationTransformer):

    _DEFAULT_VARIANCE_CUTOFF = 0.95

    def __init__(self, epsilon=None, lag=0, reversible=True, stride=1,
                 skip=0, weights=None, ncov_max=None, dim=_DEFAULT_VARIANCE_CUTOFF,
                 scaling='kinetic_map',
                 # deprecated:
                 kinetic_map=None, commute_map=None, var_cutoff=None):
        super(TICABase, self).__init__()
        self.epsilon = epsilon
        self.lag = lag
        self.reversible = reversible
        self.stride = stride
        self.skip = skip
        self.weights = weights
        self.ncov_max = ncov_max
        self.dim = dim
        import warnings
        # handle deprecated arguments
        if not (kinetic_map is None and commute_map is None):
            if kinetic_map and commute_map:
                raise ValueError('Trying to use both kinetic_map and commute_map. Use either or.')
            elif kinetic_map:
                scaling = 'kinetic_map'
            elif not kinetic_map:
                scaling = None
            elif not commute_map:
                raise
            if (kinetic_map or commute_map) and not reversible:
                warnings.warn("Cannot use kinetic_map or commute_map for non-reversible processes, both will be set to"
                              "False.")
                scaling = None

        if var_cutoff != None:
            var_cutoff = float(var_cutoff)
            warnings.warn('passed deprecated setting "var_cutoff", '
                          'will override passed "dim" ({dim}) parameter with {var_cutoff}'
                          .format(dim=dim, var_cutoff=var_cutoff))
            if var_cutoff != self._DEFAULT_VARIANCE_CUTOFF and dim != -1 and var_cutoff != 1.0:
                raise ValueError('Trying to set both the number of dimension and the subspace variance. '
                                 'Use either one or the other.')
            self.dim = var_cutoff
        if isinstance(kinetic_map, bool) and kinetic_map:
            assert scaling == 'kinetic_map'
        assert self.dim >= 0

        self.scaling = scaling

    @property
    def lag(self):
        """ lag time of correlation matrix :math:`C_{\tau}` """
        return self.model.lag

    @lag.setter
    def lag(self, new_tau):
        self.model.lag = new_tau

    @abstractproperty
    def model(self):
        raise NotImplementedError()

    @property
    def mean(self):
        """ mean of input features """
        return self.model.mean

    @mean.setter
    def mean(self, value):
        self.model.mean = value

    def dimension(self):
        """ output dimension """
        return self.model.dimension()

    def _transform_array(self, X):
        r"""Projects the data onto the dominant independent components.

        Parameters
        ----------
        X : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data
        """
        X_meanfree = X - self.mean
        Y = np.dot(X_meanfree, self.eigenvectors[:, 0:self.dimension()])

        return Y.astype(self.output_type())

    @property
    def eigenvalues(self):
        r""" Eigenvalues of the TICA problem (usually denoted :math:`\lambda`)

        Returns
        -------
        eigenvalues: 1D np.array
        """
        return self.model.eigenvalues

    @property
    def eigenvectors(self):
        r""" Eigenvectors of the TICA problem, columnwise

        Returns
        -------
        eigenvectors: (N,M) ndarray
        """
        return self.model.eigenvectors

    @property
    def cumvar(self):
        r""" Cumulative sum of the the TICA eigenvalues

        Returns
        -------
        cumvar: 1D np.array
        """
        return self.model.cumvar

    def output_type(self):
        # TODO: handle the case of conjugate pairs
        if np.all(np.isreal(self.eigenvectors[:, 0:self.dimension()])) or \
                np.allclose(np.imag(self.eigenvectors[:, 0:self.dimension()]), 0):
            return super(TICABase, self).output_type()
        else:
            return np.complex64

    @property
    def timescales(self):
        r"""Implied timescales of the TICA transformation

        For each :math:`i`-th eigenvalue, this returns

        .. math::

            t_i = -\frac{\tau}{\log(|\lambda_i|)}

        where :math:`\tau` is the :py:obj:`lag` of the TICA object and :math:`\lambda_i` is the `i`-th
        :py:obj:`eigenvalue <eigenvalues>` of the TICA object.

        Returns
        -------
        timescales: 1D np.array
            numpy array with the implied timescales. In principle, one should expect as many timescales as
            input coordinates were available. However, less eigenvalues will be returned if the TICA matrices
            were not full rank or :py:obj:`var_cutoff` was parsed
        """
        return -self.lag / np.log(np.abs(self.eigenvalues))

    @property
    def feature_TIC_correlation(self):
        r"""Instantaneous correlation matrix between mean-free input features and TICs

        Denoting the input features as :math:`X_i` and the TICs as :math:`\theta_j`, the instantaneous, linear correlation
        between them can be written as

        .. math::

            \mathbf{Corr}(X_i - \mu_i, \mathbf{\theta}_j) = \frac{1}{\sigma_{X_i - \mu_i}}\sum_l \sigma_{(X_i - \mu_i)(X_l - \mu_l} \mathbf{U}_{li}

        The matrix :math:`\mathbf{U}` is the matrix containing, as column vectors, the eigenvectors of the TICA
        generalized eigenvalue problem .

        Returns
        -------
        feature_TIC_correlation : ndarray(n,m)
            correlation matrix between input features and TICs. There is a row for each feature and a column
            for each TIC.
        """
        feature_sigma = np.sqrt(np.diag(self.cov))
        return np.dot(self.cov, self.eigenvectors[:, : self.dimension()]) / feature_sigma[:, np.newaxis]

    @property
    def cov(self):
        """ covariance matrix of input data. """
        return self.model.cov

    @cov.setter
    def cov(self, value):
        self.model.cov = value

    @property
    def cov_tau(self):
        """ covariance matrix of time-lagged input data. """
        return self.model.cov_tau

    @cov_tau.setter
    def cov_tau(self, value):
        self.model.cov_tau = value

    @property
    def dim(self):
        """output dimension (input parameter).

        Maximum number of significant independent components to use to reduce dimension of input data. -1 means
        all numerically available dimensions (see epsilon) will be used unless reduced by var_cutoff.
        Setting dim to a positive value is exclusive with var_cutoff.
        """
        return self.model.dim

    @dim.setter
    def dim(self, value):
        self.model.dim = value

    @property
    @deprecated('use dim property with a floating point value.')
    def var_cutoff(self):
        """ Kinetic variance cutoff. Deprecated, use dim property with a floating point value.

        Should be given in terms of a percentage between (0, 1.0].
        Can only be applied if dim is not set explicitly.
        """
        return self.model.dim

    @var_cutoff.setter
    @deprecated('use dim property with a floating point value.')
    def var_cutoff(self, value):
        self.model.dim = value

    @property
    def scaling(self):
        return self.model.scaling

    @scaling.setter
    def scaling(self, value):
        self.model.scaling = value

    @property
    def epsilon(self):
        return self.model.epsilon

    @epsilon.setter
    def epsilon(self, value):
        self.model.epsilon = value
