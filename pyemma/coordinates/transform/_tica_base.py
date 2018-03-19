from abc import abstractproperty

import numpy as np
from decorator import decorator
from pyemma._base.model import Model
from pyemma._base.serialization.serialization import SerializableMixIn

from pyemma.coordinates.data._base.transformer import StreamingEstimationTransformer

__author__ = 'marscher'


@decorator
def _lazy_estimation(func, *args, **kw):
    assert isinstance(args[0], TICABase)
    tica_obj = args[0]
    if not tica_obj._estimated:
        tica_obj._diagonalize()
    return func(*args, **kw)


class TICAModelBase(Model, SerializableMixIn):
    __serialize_version = 0

    def set_model_params(self, mean=None, cov_tau=None, cov=None,
                         cumvar=None, eigenvalues=None, eigenvectors=None):
        self.update_model_params(cov=cov, cov_tau=cov_tau,
                                 mean=mean, cumvar=cumvar,
                                 eigenvalues=eigenvalues,
                                 eigenvectors=eigenvectors)


class TICABase(StreamingEstimationTransformer):

    _DEFAULT_VARIANCE_CUTOFF = 0.95

    @property
    def lag(self):
        """ lag time of correlation matrix :math:`C_{\tau}` """
        return self._lag

    @lag.setter
    def lag(self, new_tau):
        self._lag = new_tau

    @property
    def dim(self):
        """output dimension (input parameter).

        Maximum number of significant independent components to use to reduce dimension of input data. -1 means
        all numerically available dimensions (see epsilon) will be used unless reduced by var_cutoff.
        Setting dim to a positive value is exclusive with var_cutoff.
        """
        return self._dim

    @dim.setter
    def dim(self, value):
        if int(value) > -1:
            self._var_cutoff = 1.0
        self._dim = int(value)

    @property
    def var_cutoff(self):
        """ Kinetic variance cutoff

        Should be given in terms of a percentage between (0, 1.0].
        Can only be applied if dim is not set explicitly.
        """
        return self._var_cutoff

    @var_cutoff.setter
    def var_cutoff(self, value):
        v = float(value)
        if not hasattr(self, '_dim'):
            raise RuntimeError('need to set dim before var_cutoff')

        if not (0 < v <= 1.0):
            raise ValueError('variance cutoff has to be in interval (0, 1.0]')

        if v != TICABase._DEFAULT_VARIANCE_CUTOFF and self.dim != -1 and v != 1.0:
            raise ValueError('Trying to set both the number of dimension and the subspace variance. '
                             'Use either one or the other.')
        self._var_cutoff = v

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
        if self.dim > -1:
            return self.dim
        d = None
        if self.dim != -1 and not self._estimated:  # fixed parametrization
            d = self.dim
        elif self._estimated:  # parametrization finished. Dimension is known
            dim = len(self.eigenvalues)
            if self.var_cutoff < 1.0:  # if subspace_variance, reduce the output dimension if needed
                dim = min(dim, np.searchsorted(self.cumvar, self.var_cutoff) + 1)
            d = dim
        elif self.var_cutoff == 1.0:  # We only know that all dimensions are wanted, so return input dim
            d = self.data_producer.dimension()
        else:  # We know nothing. Give up
            raise RuntimeError('Requested dimension, but the dimension depends on the cumulative variance and the '
                               'transformer has not yet been estimated. Call estimate() before.')
        return d

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
    @_lazy_estimation
    def eigenvalues(self):
        r""" Eigenvalues of the TICA problem (usually denoted :math:`\lambda`)

        Returns
        -------
        eigenvalues: 1D np.array
        """
        return self.model.eigenvalues

    @property
    @_lazy_estimation
    def eigenvectors(self):
        r""" Eigenvectors of the TICA problem, columnwise

        Returns
        -------
        eigenvectors: (N,M) ndarray
        """
        return self.model.eigenvectors

    @property
    @_lazy_estimation
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
    @_lazy_estimation
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
    @_lazy_estimation
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
