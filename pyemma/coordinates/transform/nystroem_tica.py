# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import absolute_import

import numpy as np
from decorator import decorator

from pyemma._base.model import Model
from pyemma._ext.variational.solvers.direct import sort_by_norm, spd_inv_split, eig_corr
from pyemma._ext.variational.util import ZeroRankError
from pyemma.coordinates.data._base.transformer import StreamingEstimationTransformer
from pyemma.coordinates.estimation.covariance import LaggedCovariance
from pyemma.util.annotators import fix_docs
from pyemma.util.reflection import get_default_args
import warnings
import copy

__all__ = ['NystroemTICA']

__author__ = 'litzinger'


class NystroemTICAModel(Model):
    def set_model_params(self, mean, cov, cov_tau, diag, column_indices):
        self.mean = mean
        self.cov = cov
        self.cov_tau = cov_tau
        self.diag = diag
        self.column_indices = column_indices

@decorator
def _lazy_estimation(func, *args, **kw):
    assert isinstance(args[0], NystroemTICA)
    tica_obj = args[0]
    if not tica_obj._estimated:
        tica_obj._diagonalize()
    return func(*args, **kw)


@fix_docs
class NystroemTICA(StreamingEstimationTransformer):
    r""" Sparse sampling implementation of time-lagged independent component analysis (TICA)"""

    def __init__(self, lag, max_columns,
                 dim=-1, var_cutoff=0.95, epsilon=1e-6,
                 stride=1, skip=0, reversible=True, ncov_max=float('inf'),
                 initial_columns=None, nsel=1, selection_strategy='spectral-oasis', neig=None):
        r""" Sparse sampling implementation [1]_ of time-lagged independent component analysis (TICA) [2]_, [3]_, [4]_.

        Parameters
        ----------
        lag : int
            lag time
        max_columns : int
            Maximum number of columns (features) to use in the approximation.
        dim : int, optional, default -1
            Maximum number of significant independent components to use to reduce dimension of input data. -1 means
            all numerically available dimensions (see epsilon) will be used unless reduced by var_cutoff.
            Setting dim to a positive value is exclusive with var_cutoff.
        var_cutoff : float in the range [0,1], optional, default 0.95
            Determines the number of output dimensions by including dimensions until their cumulative kinetic variance
            exceeds the fraction subspace_variance. var_cutoff=1.0 means all numerically available dimensions
            (see epsilon) will be used, unless set by dim. Setting var_cutoff smaller than 1.0 is exclusive with dim.
        epsilon : float, optional, default 1e-6
            Eigenvalue norm cutoff. Eigenvalues of :math:`C_0` with norms <= epsilon will be
            cut off. The remaining number of eigenvalues define the size
            of the output.
        stride: int, optional, default 1
            Use only every stride-th time step. By default, every time step is used.
        skip : int, optional, default 0
            Skip the first initial n frames per trajectory.
        reversible: bool, optional, default True
            Symmetrize correlation matrices :math:`C_0`, :math:`C_{\tau}`.
        initial_columns : list, ndarray(k, dtype=int), int, or None, optional, default None
            Columns used for an initial approximation. If a list or an 1-d ndarray
            of integers is given, use these column indices. If an integer is given,
            use that number of randomly selected indices. If None is given, use
            one randomly selected column.
        nsel : int, optional, default 1
            Number of columns to select and add per iteration and pass through the data.
            Larger values provide for better pass-efficiency.
        selection_strategy : str, optional, default 'spectral-oasis'
            Strategy to use for selecting new columns for the approximation.
            Can be 'random', 'oasis' or 'spectral-oasis'.
        neig : int or None, optional, default None
            Number of eigenvalues to be optimized by the selection process.
            If None, use the whole available eigenspace

        Notes
        -----
        Perform a sparse approximation of time-lagged independent component analysis (TICA)
        :class:`TICA <pyemma.coordinates.transform.TICA>`. The starting point is the
        generalized eigenvalue problem

        .. math:: C_{\tau} r_i = C_0 \lambda_i(\tau) r_i.

        Instead of computing the full matrices involved in this problem, we conduct
        a Nyström approximation [5]_ of the matrix :math:`C_0` by means of the
        accelerated sequential incoherence selection (oASIS) algorithm [6]_ and,
        in particular, its extension called spectral oASIS [1]_.

        Iteratively, we select a small number of columns such that the resulting
        Nyström approximation is sufficiently accurate. This selection represents
        in turn a subset of important features, for which we obtain a generalized
        eigenvalue problem similar to the one above, but much smaller in size.
        Its generalized eigenvalues and eigenvectors provide an approximation
        to those of the full TICA solution [1]_.

        References
        ----------
        .. [1] F. Litzinger, L. Boninsegna, H. Wu, F. Nüske, R. Patel, R. Baraniuk, F. Noé, and C. Clementi.
           Rapid calculation of molecular kinetics using compressed sensing (2018). (submitted)
        .. [2] Perez-Hernandez G, F Paul, T Giorgino, G De Fabritiis and F Noe. 2013.
           Identification of slow molecular order parameters for Markov model construction
           J. Chem. Phys. 139, 015102. doi:10.1063/1.4811489
        .. [3] Schwantes C, V S Pande. 2013.
           Improvements in Markov State Model Construction Reveal Many Non-Native Interactions in the Folding of NTL9
           J. Chem. Theory. Comput. 9, 2000-2009. doi:10.1021/ct300878a
        .. [4] L. Molgedey and H. G. Schuster. 1994.
           Separation of a mixture of independent signals using time delayed correlations
           Phys. Rev. Lett. 72, 3634.
        .. [5] P. Drineas and M. W. Mahoney.
           On the Nystrom method for approximating a Gram matrix for improved kernel-based learning.
           Journal of Machine Learning Research, 6:2153-2175 (2005).
        .. [6] Raajen Patel, Thomas A. Goldstein, Eva L. Dyer, Azalia Mirhoseini, Richard G. Baraniuk.
           oASIS: Adaptive Column Sampling for Kernel Matrix Approximation.
           arXiv: 1505.05208 [stat.ML].

        """
        default_var_cutoff = get_default_args(self.__init__)['var_cutoff']
        if dim != -1 and var_cutoff != default_var_cutoff:
            raise ValueError('Trying to set both the number of dimension and the subspace variance. Use either one or the other.')
        super(NystroemTICA, self).__init__()

        if dim > -1:
            var_cutoff = 1.0

        if initial_columns is None:
            initial_columns = 1
        if isinstance(initial_columns, int):
            i = initial_columns
            initial_columns = lambda N: np.random.choice(N, i, replace=False)

        self._covar = LaggedCovariance(c00=True, c0t=True, ctt=False, remove_data_mean=True, reversible=reversible,
                                       lag=lag, bessel=False, stride=stride, skip=skip, ncov_max=ncov_max)
        self._diag = LaggedCovariance(c00=True, c0t=True, ctt=False, remove_data_mean=True, reversible=reversible,
                                      lag=lag, bessel=False, stride=stride, skip=skip, ncov_max=ncov_max,
                                      diag_only=True)
        self._oasis = None

        # empty dummy model instance
        self._model = NystroemTICAModel()
        self.set_params(lag=lag, max_columns=max_columns,
                        dim=dim, var_cutoff=var_cutoff,
                        epsilon=epsilon, reversible=reversible, stride=stride, skip=skip,
                        ncov_max=ncov_max,
                        initial_columns=initial_columns, nsel=nsel, selection_strategy=selection_strategy, neig=neig)

    @property
    def lag(self):
        """ lag time of correlation matrix :math:`C_{\tau}` """
        return self._lag

    @lag.setter
    def lag(self, new_tau):
        self._lag = new_tau

    def describe(self):
        try:
            dim = self.dimension()
        except AttributeError:
            dim = self.dim
        return "[NystroemTICA, lag = %i; max. columns = %i; max. output dim. = %i]" % (self._lag, self._max_columns, dim)

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

    @property
    def mean(self):
        """ mean of input features """
        return self._model.mean

    @mean.setter
    def mean(self, value):
        self._model.mean = value

    def estimate(self, X, **kwargs):
        r"""
        Chunk-based parameterization of NystroemTICA.
        Iterates over all data several times to select important columns and
        estimate the mean, covariance and time-lagged covariance. Finally, the
        small-scale generalized eigenvalue problem is solved to determine
        the approximate independent components.
        """
        return super(NystroemTICA, self).estimate(X, **kwargs)

    def _estimate(self, iterable, **kw):
        from pyemma.coordinates.data import DataInMemory
        if not isinstance(iterable, DataInMemory):
            self._logger.warning('Every iteration of the selection process involves streaming of all data and featurization. '+
                                 'Depending on your setup, this might be inefficient.')

        indim = iterable.dimension()
        if not self.dim <= indim:
            raise RuntimeError("requested more output dimensions (%i) than dimension"
                               " of input data (%i)" % (self.dim, indim))

        if callable(self.initial_columns):
            self.initial_columns = self.initial_columns(indim)
        if not len(np.array(self.initial_columns).shape) == 1:
            raise ValueError('initial_columns must be either None, an integer, a list, or a 1-d numpy array.')

        self._diag.estimate(iterable, **kw)

        self._covar.column_selection = self.initial_columns
        self._covar.estimate(iterable, **kw)
        self._model.update_model_params(cov_tau=self._covar.cov_tau)

        self._oasis = oASIS_Nystroem(self._diag.cov, self._covar.cov, self.initial_columns)
        self._oasis.set_selection_strategy(strategy=self.selection_strategy, nsel=self.nsel, neig=self.neig)

        while len(self._oasis.column_indices) < self.max_columns:
            cols = self._oasis.select_columns()
            if cols is None:
                break
            if len(cols) == 0 or np.all(np.in1d(cols, self._oasis.column_indices)):
                self._logger.warning("Iteration ended prematurely: No more columns to select.")
                break
            self._covar.column_selection = cols
            self._covar.estimate(iterable, **kw)
            ix = self._oasis.add_columns(self._covar.cov, cols)
            ix = np.in1d(cols, ix)
            if np.any(ix):
                added_columns = self._covar.cov_tau[:, ix]
                self._model.update_model_params(cov_tau=np.concatenate((self._model.cov_tau, added_columns), axis=1))

        self._model.update_model_params(mean=self._covar.mean,
                                        diag=self._diag.cov,
                                        cov=self._oasis.Ck,
                                        column_indices=self._oasis.column_indices)
        self._diagonalize()

        return self._model

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

    def _diagonalize(self):
        # diagonalize with low rank approximation
        self._logger.debug("Diagonalize Cov and Cov_tau.")
        Wktau = self._model.cov_tau[self._model.column_indices, :].copy()
        try:
            eigenvalues, eigenvectors = eig_corr(self._oasis.Wk, Wktau, self.epsilon, sign_maxelement=True)
        except ZeroRankError:
            raise ZeroRankError('All input features are constant in all time steps. No dimension would be left after dimension reduction.')
        self._logger.debug("Finished diagonalization.")

        # compute cumulative variance
        cumvar = np.cumsum(np.abs(eigenvalues) ** 2)
        cumvar /= cumvar[-1]

        self._model.update_model_params(cumvar=cumvar,
                                        eigenvalues=eigenvalues,
                                        eigenvectors=eigenvectors)

        self._estimated = True

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
    def cov(self):
        """ covariance matrix of input data. """
        return self._model.cov

    @cov.setter
    def cov(self, value):
        self._model.cov = value

    @property
    def cov_tau(self):
        """ covariance matrix of time-lagged input data. """
        return self._model.cov_tau

    @cov_tau.setter
    def cov_tau(self, value):
        self._model.cov_tau = value

    @property
    def column_indices(self):
        """ Indices of columns used in the approximation. """
        return self._model.column_indices

    @property
    @_lazy_estimation
    def eigenvalues(self):
        r""" Eigenvalues of the TICA problem (usually denoted :math:`\lambda`)

        Returns
        -------
        eigenvalues: 1D np.array
        """
        return self._model.eigenvalues

    @property
    @_lazy_estimation
    def eigenvectors(self):
        r""" Eigenvectors of the TICA problem, columnwise

        Returns
        -------
        eigenvectors: (N,M) ndarray
        """
        return self._model.eigenvectors

    @property
    @_lazy_estimation
    def cumvar(self):
        r""" Cumulative sum of the the TICA eigenvalues

        Returns
        -------
        cumvar: 1D np.array
        """
        return self._model.cumvar

    def output_type(self):
        # TODO: handle the case of conjugate pairs
        if np.all(np.isreal(self.eigenvectors[:, 0:self.dimension()])) or \
                np.allclose(np.imag(self.eigenvectors[:, 0:self.dimension()]), 0):
            return super(NystroemTICA, self).output_type()
        else:
            return np.complex64


class oASIS_Nystroem:
    r""" Implements a sparse sampling method for very large symmetric matrices.

    The aim of this method is to provide a low-rank approximation of a very large symmetric matrix
    $C \in \mathbb{R}^{n \times n}$.
    This method uses the Nystroem approximation [1]_ and the Accelerated Sequential Incoherent Selection (oASIS)
    method [2]_ for selecting a small set of $m$ column vectors to achieve a sparse approximation of $C$.
    Using an extended method called spectral oASIS [3]_, multiple columns can be selected at once,
    thereby reducing the required number of passes through the data.

    While the original basis set size can be very large (e.g. millions), $m$ is much smaller (e.g. hundreds or less).
    The method never computes the large $n \times n$ matrices explicitly, but at most evaluates matrices of the
    size $n \times m$.

    Example
    -------

    In this example we attempt to approximate a correlation matrix from random data:

    >>> # generate random time series
    >>> import numpy as np
    >>> X = np.random.randn((10000,100))
    >>> # compute full correlation matrix
    >>> C0 = np.dot(X.T, X)

    We will compute the full correlation matrix as a reference, and start oASIS with 3 out of 100 columns:

    >>> # approximate correlation matrix
    >>> d = np.diag(C0)
    >>> cols = [0,49,99]
    >>> C0_k = C0[:,cols]
    >>> oasis = oASIS_Nystroem(np.diag(C0), C0_k, cols)
    >>> # show error of the current approximation
    >>> print np.max(oasis.error)

    Now we conduct the approximation. We ask oASIS which columns should be computed next, compute them with whichever
    algorithm applies, and update the oASIS approximation. This can be repeated until the error is small enough or
    until a certain number of columns is reached.

    >>> # ask oASIS which column we should compute next
    >>> newcol = oasis.select_columns()
    >>> # recompute the new column yourself
    >>> c = np.dot(X.T, X[:,newcol][:,None])
    >>> # update oASIS
    >>> oasis.add_column(c, newcol)
    >>> # take note of the new column index
    >>> cols.append(newcol)
    >>> # show error of the current approximation
    >>> print np.max(oasis.error)

    References
    ----------
    .. [1] P. Drineas and M. W. Mahoney.
       On the Nystrom method for approximating a Gram matrix for improved kernel-based learning.
       Journal of Machine Learning Research, 6:2153-2175 (2005).
    .. [2] Raajen Patel, Thomas A. Goldstein, Eva L. Dyer, Azalia Mirhoseini, Richard G. Baraniuk.
       oASIS: Adaptive Column Sampling for Kernel Matrix Approximation.
       arXiv: 1505.05208 [stat.ML].
    .. [3] F. Litzinger, L. Boninsegna, H. Wu, F. Nüske, R. Patel, R. Baraniuk, F. Noé, and C. Clementi.
       Rapid calculation of molecular kinetics using compressed sensing (2018). (submitted)

    """

    def __init__(self, d, C_k, columns):
        """
        Initializes the oASIS_Nystroem method.

        Parameters
        ----------
        d : ndarray((n,), dtype=float)
            diagonal (autocorrelation) elements of :math:`A`
        C_k : ndarray((n,k), dtype=float)
            column matrix with the k selected columns of the spd matrix :math:`A`.
        columns : ndarray((k,), dtype=int) or list of ints of size :math:`k`
            array of selected column indices (such that formally :math:`A_k = \mathrm{A[:,columns]}` )

        """
        # store copy of diagonals
        self._d = np.array(d)
        # store copy of column submatrix
        self._C_k = np.array(C_k)
        # store copy of pre-selected columns
        self._columns = np.array(columns)
        # number of pre-selected columns
        self._k = self._columns.shape[0]
        # total size
        self._n = self._C_k.shape[0]

        self.update_inverse()

        # update error
        self._compute_error()

        # set default selection strategy
        self._selection_strategy = selection_strategy(oasis_obj=self, strategy='spectral-oasis', nsel=len(columns))

    @property
    def n(self):
        """ Total number of rows (large matrix size) """
        return self._n

    @property
    def k(self):
        """ Current number of columns """
        return self._k

    @property
    def Ck(self):
        """ The submatrix of selected columns """
        return self._C_k

    @property
    def Wk(self):
        """ The block matrix selected by rows and columns """
        return self._C_k[self._columns, :]

    @property
    def Wk_inv(self):
        """ The inverse of the block matrix selected by rows and columns """
        return self._W_k_inv

    @property
    def diag(self):
        """ The diagonal """
        return self._d

    @property
    def column_indices(self):
        """ The selected column indices """
        return np.array(self._columns)

    def _compute_error(self):
        """ Evaluate the absolute error of the Nystroem approximation for each column """
        # evaluate error of Nystroem approx for each new column
        # err_i = sum_j R_{k,ij} A_{k,ji} - d_i
        self._err = np.sum(np.multiply(self._R_k,self._C_k.T),axis=0) - self._d

    @property
    def error(self):
        """ Absolute error of the Nystroem approximation by column """
        return self._err

    def set_selection_strategy(self, strategy='spectral-oasis', nsel=1, neig=None):
        """ Defines the column selection strategy

        Parameters
        ----------
        strategy : str
            One of the following strategies to select new columns:
            random : randomly choose from non-selected columns
            oasis : maximal approximation error in the diagonal of :math:`A`
            spectral-oasis : selects the nsel columns that are most distanced in the oASIS-error-scaled dominant eigenspace
        nsel : int
            number of columns to be selected in each round
        neig : int or None, optional, default None
            Number of eigenvalues to be optimized by the selection process.
            If None, use the whole available eigenspace

        """
        self._selection_strategy = selection_strategy(self, strategy, nsel, neig)

    def select_columns(self):
        """ Selects new columns of :math:`A` using the given selection strategy

        Returns
        -------
        selected_columns : ndarray((ncols), dtype=int)
            An array containing the selected column indices

        """
        return self._selection_strategy.select()

    def update_inverse(self):
        """ Recomputes W_k_inv and R_k given the current column selection

        When computed, the block matrix inverse W_k_inv will be updated. This is useful when you want to compute
        eigenvalues or get an approximation for the full matrix or individual columns.
        Calling this function is not strictly necessary, but then you rely on the fact that the updates did not
        accumulate large errors. That depends very much on how columns were added. Adding columns with very small
        Schur complement causes accumulation of errors and is more likely to make it necessary to update the inverse.

        """
        # compute R_k and W_k_inv
        Wk = self._C_k[self._columns, :]
        self._W_k_inv = np.linalg.pinv(Wk)
        self._R_k = np.dot(self._W_k_inv, self._C_k.T)

    def add_column(self, col, icol, update_error=True):
        """ Attempts to add a single column of :math:`A` to the Nystroem approximation and updates the local matrices

        Parameters
        ----------
        col : ndarray((N,), dtype=float)
            new column of :math:`A`
        icol : int
            index of new column within :math:`A`
        update_error : bool, optional, default = True
            If True, the absolute and relative approximation error will be updated after adding the column.
            If False, then not.

        Return
        ------
        success : bool
            True if the new column was added to the approximation. False if not.

        """
        # copy column
        col = copy.deepcopy(col)

        # convenience access
        k = self._k
        d = self._d
        R = self._R_k
        Winv = self._W_k_inv

        b_new = col[self._columns][:, None]
        d_new = d[icol]
        q_new = R[:, icol][:, None]

        # calculate R_new
        schur_complement = d_new - np.dot(b_new.T, q_new)  # Schur complement
        if np.isclose(schur_complement, 0):
            return False

        # otherwise complete the update
        s_new = 1./schur_complement
        qC = np.dot(b_new.T, R)

        # update Winv
        Winv_new = np.zeros((k+1, k+1))
        Winv_new[0:k, 0:k] = Winv+s_new*np.dot(q_new, q_new.T)
        Winv_new[0:k, k] = -s_new*q_new[0:k, 0]
        Winv_new[k, 0:k] = -s_new*q_new[0:k, 0].T
        Winv_new[k, k] = s_new

        R_new = np.vstack((R + s_new * np.dot(q_new, (qC - col.T)), s_new*(-qC + col.T)))

        # forcing known structure on R_new
        sel_new = np.append(self._columns, icol)
        R_new[:, sel_new] = np.eye(k+1)

        # update Winv
        self._W_k_inv = Winv_new
        # update R
        self._R_k = R_new
        # update C0_k
        self._C_k = np.hstack((self._C_k, col[:, None]))
        # update number of selected columns
        self._k += 1
        # add column to present selection
        self._columns = np.append(self._columns, icol)

        # update error
        if update_error:
            self._compute_error()

        # exit with success
        return True

    def add_columns(self, C_k_new, columns_new):
        r""" Attempts to adds a set of new columns of :math:`A` to the Nystroem approximation and updates the local matrices

        Parameters
        ----------
        C_k_new : ndarray((N,k), dtype=float)
            :math:`k` new columns of :math:`A`
        columns_new : int
            indices of new columns within :math:`A`, in the same order as the C_k_new columns

        Return
        ------
        cols_added : ndarray of int
            Columns that were added successfully. Columns are only added when their Schur complement exceeds 0,
            which is normally true for columns that were not yet added, but the Schur complement may become 0 even
            for new columns as a result of numerical cancellation errors.

        """
        added = []
        for (i, c) in enumerate(columns_new):
            if self.add_column(C_k_new[:, i], c, update_error=False):
                added.append(c)
        # update error only once
        self._compute_error()
        # return the columns that were successfully added
        return np.array(added)

    def approximate_matrix(self):
        r""" Computes the Nystroem approximation of the matrix $A \in \mathbb{R}^{n \times n}$.

        WARNING: This will attempt to construct a $n \times n$ matrix. The computation effort of doing this is
        $O(m n^2)$, and the memory requirements can be huge for large $n$. So be sure you know what you are asking
        for when calling this method.

        """
        C_approx = np.dot(self._C_k, self._R_k)
        return C_approx

    def approximate_column(self, i):
        """ Computes the Nystroem approximation of column :math:`i` of matrix $A \in \mathbb{R}^{n \times n}$.

        """
        col_approx = np.dot(self._C_k, self._R_k[:, i])
        return col_approx

    def approximate_cholesky(self, epsilon=1e-6):
        r""" Compute low-rank approximation to the Cholesky decomposition of target matrix.

        The decomposition will be conducted while ensuring that the spectrum of `A_k^{-1}` is positive.

        Parameters
        ----------
        epsilon : float, optional, default 1e-6
            If truncate=True, this determines the cutoff for eigenvalue norms. If negative eigenvalues occur,
            with larger norms than epsilon, the largest negative eigenvalue norm will be used instead of epsilon, i.e.
            a band including all negative eigenvalues will be cut off.

        Returns
        -------
        L : ndarray((n,m), dtype=float)
            Cholesky matrix such that `A \approx L L^{\top}`. Number of columns :math:`m` is most at the number of columns
            used in the Nystroem approximation, but may be smaller if truncate=True.

        """
        # compute the Eigenvalues of C0 using Schur factorization
        Wk = self._C_k[self._columns, :]
        L0 = spd_inv_split(Wk)
        L = np.dot(self._C_k, L0)

        return L

    def approximate_eig(self, epsilon=1e-6):
        """ Compute low-rank approximation of the eigenvalue decomposition of target matrix.

        If spd is True, the decomposition will be conducted while ensuring that the spectrum of `A_k^{-1}` is positive.

        Parameters
        ----------
        epsilon : float, optional, default 1e-6
            If truncate=True, this determines the cutoff for eigenvalue norms. If negative eigenvalues occur,
            with larger norms than epsilon, the largest negative eigenvalue norm will be used instead of epsilon, i.e.
            a band including all negative eigenvalues will be cut off.

        Returns
        -------
        s : ndarray((m,), dtype=float)
            approximated eigenvalues. Number of eigenvalues returned is at most the number of columns used in the
            Nystroem approximation, but may be smaller if truncate=True.

        W : ndarray((n,m), dtype=float)
            approximated eigenvectors in columns. Number of eigenvectors returned is at most the number of columns
            used in the Nystroem approximation, but may be smaller if truncate=True.

        """
        L = self.approximate_cholesky(epsilon=epsilon)
        LL = np.dot(L.T, L)
        s, V = np.linalg.eigh(LL)
        # sort
        s, V = sort_by_norm(s, V)

        # back-transform eigenvectors
        Linv = np.linalg.pinv(L.T)
        V = np.dot(Linv, V)

        # normalize eigenvectors
        ncol = V.shape[1]
        for i in range(ncol):
            if not np.allclose(V[:, i], 0):
                V[:, i] /= np.sqrt(np.dot(V[:, i], V[:, i]))

        return s, V


class SelectionStrategy:
    def __init__(self, oasis_obj, strategy='spectral-oasis', nsel=1, neig=None):
        """ Abstract selection strategy class

        Parameters
        ----------
        oasis_obj : oASIS_Nystroem
            The associated oASIS_Nystroem object.
        strategy : str, optional, default 'spectral-oasis'
            One of the following strategies to select new columns:
            random : randomly choose from non-selected columns
            oasis : maximal approximation error in the diagonal of :math:`A`
            spectral-oasis : selects the nsel columns that are most distance in the oASIS-error-scaled dominant eigenspace
        nsel : int, optional, default 1
            Number of columns to be selected in each round
        neig : int or None, optional, default None
            Number of eigenvalues to be optimized by the selection process. Only used in 'spectral-oasis'.
            If None, use the whole available eigenspace

        """
        self._oasis_obj = oasis_obj
        self._strategy = strategy
        self._nsel = nsel
        self._neig = neig

    @property
    def strategy(self):
        return self._strategy

    @property
    def nsel(self):
        return self._nsel

    def _check_nsel(self):
        if self._nsel > self._oasis_obj._n - self._oasis_obj._k:
            # less columns left than requested
            ncols = self._oasis_obj._n - self._oasis_obj._k
            warnings.warn('Requested more columns than are left to select. Returning only '+str(ncols)+' columns.')
            return ncols
        # nothing left to select?
        if self._oasis_obj._n == self._oasis_obj._k:
            warnings.warn('Requested more columns but there are none left. Returning None.')
            return None
        return self._nsel

    def select(self):
        """ Selects next column indexes according to defined strategy

        Returns
        -------
        cols : ndarray((nsel,), dtype=int)
            selected columns

        """
        err = self._oasis_obj.error
        # check nsel
        nsel = self._check_nsel()
        if nsel is None:
            return None
        # go
        return self._select(nsel, err)

    def _select(self, nsel, err):
        """ Override me to do selection """
        pass


class SelectionStrategyRandom(SelectionStrategy):
    """ Selects nsel random columns not yet included in the approximation """
    def _select(self, nsel, err):
        # do select
        sel = []
        while len(sel) < nsel:
            i = np.random.choice(self._oasis_obj._n)
            if not (i in sel or i in self._oasis_obj._columns):
                sel.append(i)
        return np.array(sel, dtype=int)


class SelectionStrategyOasis(SelectionStrategy):
    """ Selects the nsel columns with the largest oASIS error """
    def _select(self, nsel, err):
        return np.argsort(np.abs(err))[::-1][:nsel]


class SelectionStrategySpectralOasis(SelectionStrategy):
    """ Selects the nsel columns that are most distanced in the oASIS-error-scaled dominant eigenspace """
    def _select(self, nsel, err):
        # if one column is wanted, then we take the one with the largest error
        if nsel == 1:
            return np.array([np.argmax(np.abs(err))])
        # compute approximate eigenvectors
        _, evec = self._oasis_obj.approximate_eig()
        evec = self._fix_constant_evec(evec)
        if self._neig is None:
            neig = evec.shape[1]
        else:
            neig = min(self._neig, evec.shape[1])
        # compute eigenvector-weighted error vectors
        W = err[:, None] * evec[:, :neig]
        # initialize selection
        sel = np.zeros(nsel, dtype=int)
        n = np.shape(W)[0]
        # look for the most distant point from 0
        d_to_0 = np.linalg.norm(W, axis=1)
        sel[0] = np.argmax(d_to_0)
        # append most distance point
        for i in range(1, nsel):
            d_to_i = d_to_0[:]
            for k in range(nsel):
                d_to_i = np.minimum(d_to_i, np.linalg.norm(W - W[sel[k], :], axis=1))
            sel[i] = np.argmax(d_to_i)
        return np.unique(sel)

    def _fix_constant_evec(self, evecs):
        # test if first vector is trying to approximate constant vector
        evec0 = evecs[:, 0] / np.max(evecs[:, 0])  # make sure the vector has positive elements.
        if np.min(evec0) > -1e-10:  # this looks like a constant eigenvector
            evecs[:, 0] = 1  # make it really constant
        else:  # we don't have it, so add it
            evecs = np.hstack([np.ones((evecs.shape[0], 1)), evecs])
        return evecs


def selection_strategy(oasis_obj, strategy='spectral-oasis', nsel=1, neig=None):
    """ Factory for selection strategy object

    Returns
    -------
    selstr : SelectionStrategy
        Selection strategy object

    """
    strategy = strategy.lower()
    if strategy == 'random':
        return SelectionStrategyRandom(oasis_obj, strategy, nsel=nsel, neig=neig)
    elif strategy == 'oasis':
        return SelectionStrategyOasis(oasis_obj, strategy, nsel=nsel, neig=neig)
    elif strategy == 'spectral-oasis':
        return SelectionStrategySpectralOasis(oasis_obj, strategy, nsel=nsel, neig=neig)
    else:
        raise ValueError('Selected strategy is unknown: '+str(strategy))
