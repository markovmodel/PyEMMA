# coding=utf-8

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


from types import FunctionType

import numpy as np
from pyemma.util.types import ensure_int_vector

from pyemma._base.serialization.serialization import SerializableMixIn
from pyemma._ext.variational.solvers.direct import sort_by_norm, spd_inv_split, eig_corr
from pyemma._ext.variational.util import ZeroRankError
from pyemma.coordinates.estimation.covariance import LaggedCovariance
from pyemma.coordinates.transform._tica_base import TICABase, TICAModelBase
from pyemma.util.annotators import fix_docs
from pyemma.util.reflection import get_default_args
import warnings

__all__ = ['NystroemTICA']

__author__ = 'litzinger'


class NystroemTICAModel(TICAModelBase):
    __serialize_version = 0

    def set_model_params(self, mean, cov, cov_tau, diag, column_indices, cumvar=None):
        super(NystroemTICAModel, self).set_model_params(mean=mean, cov=cov, cov_tau=cov_tau)
        self.cumvar = cumvar
        self.diag = diag
        self.column_indices = column_indices


@fix_docs
class NystroemTICA(TICABase, SerializableMixIn):
    r""" Sparse sampling implementation of time-lagged independent component analysis (TICA)"""
    __serialize_version = 0
    __serialize_fields = ()

    def __init__(self, lag, max_columns,
                 dim=-1, var_cutoff=TICABase._DEFAULT_VARIANCE_CUTOFF, epsilon=1e-6,
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
        super(NystroemTICA, self).__init__()

        self._covar = LaggedCovariance(c00=True, c0t=True, ctt=False, remove_data_mean=True, reversible=reversible,
                                       lag=lag, bessel=False, stride=stride, skip=skip, ncov_max=ncov_max)
        self._diag = LaggedCovariance(c00=True, c0t=True, ctt=False, remove_data_mean=True, reversible=reversible,
                                      lag=lag, bessel=False, stride=stride, skip=skip, ncov_max=ncov_max,
                                      diag_only=True)
        self._oasis = None

        self.dim = dim
        self.var_cutoff = var_cutoff

        self.set_params(lag=lag, max_columns=max_columns,
                        epsilon=epsilon, reversible=reversible, stride=stride, skip=skip,
                        ncov_max=ncov_max,
                        initial_columns=initial_columns, nsel=nsel, selection_strategy=selection_strategy, neig=neig)

    @property
    def model(self):
        if not hasattr(self, '_model') or self._model is None:
            self._model = NystroemTICAModel()
        return self._model

    @property
    def initial_columns(self):
        return self._initial_columns

    @initial_columns.setter
    def initial_columns(self, initial_columns):
        if not (initial_columns is None
                or isinstance(initial_columns, (int, FunctionType, np.ndarray))):
            raise ValueError('initial_columns has to be one of these types (None, int, function, ndarray),'
                             'but was {}'.format(type(initial_columns)))
        if initial_columns is None:
            initial_columns = 1
        if isinstance(initial_columns, int):
            i = initial_columns
            initial_columns = lambda N: np.random.choice(N, i, replace=False)
        if isinstance(initial_columns, np.ndarray):
            initial_columns = ensure_int_vector(initial_columns)
        self._initial_columns = initial_columns

    def describe(self):
        try:
            dim = self.dimension()
        except RuntimeError:
            dim = self.dim
        return "[NystroemTICA, lag = %i; max. columns = %i; max. output dim. = %i]" % (self.lag, self.max_columns, dim)

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
            self.logger.warning('Every iteration of the selection process involves streaming of all data and featurization. '
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
        self.model.update_model_params(cov_tau=self._covar.C0t_)

        self._oasis = oASIS_Nystroem(self._diag.C00_, self._covar.C00_, self.initial_columns)
        self._oasis.set_selection_strategy(strategy=self.selection_strategy, nsel=self.nsel, neig=self.neig)

        while self._oasis.k < np.min((self.max_columns, self._oasis.n)):
            cols = self._oasis.select_columns()
            if cols is None or len(cols) == 0 or np.all(np.in1d(cols, self._oasis.column_indices)):
                self.logger.warning("Iteration ended prematurely: No more columns to select.")
                break
            self._covar.column_selection = cols
            self._covar.estimate(iterable, **kw)
            ix = self._oasis.add_columns(self._covar.C00_, cols)
            ix = np.in1d(cols, ix)
            if np.any(ix):
                added_columns = self._covar.C0t_[:, ix]
                self.model.update_model_params(cov_tau=np.concatenate((self._model.cov_tau, added_columns), axis=1))

        self.model.update_model_params(mean=self._covar.mean,
                                        diag=self._diag.C00_,
                                        cov=self._oasis.Ck,
                                        column_indices=self._oasis.column_indices)
        self._diagonalize()

        return self.model

    def _diagonalize(self):
        # diagonalize with low rank approximation
        self.logger.debug("Diagonalize Cov and Cov_tau.")
        Wktau = self._model.cov_tau[self._model.column_indices, :]
        try:
            eigenvalues, eigenvectors = eig_corr(self._oasis.Wk, Wktau, self.epsilon, sign_maxelement=True)
        except ZeroRankError:
            raise ZeroRankError('All input features are constant in all time steps. '
                                'No dimension would be left after dimension reduction.')
        self.logger.debug("Finished diagonalization.")

        # compute cumulative variance
        cumvar = np.cumsum(np.abs(eigenvalues) ** 2)
        cumvar /= cumvar[-1]

        self._model.update_model_params(cumvar=cumvar,
                                        eigenvalues=eigenvalues,
                                        eigenvectors=eigenvectors)

        self._estimated = True

    @property
    def column_indices(self):
        """ Indices of columns used in the approximation. """
        return self.model.column_indices


class oASIS_Nystroem(object):
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
    >>> from numpy.random import RandomState
    >>> prng = RandomState(0)
    >>> X = np.ones((10000, 10))
    >>> X[:, :5] = prng.randn(10000, 5)
    >>> # compute full correlation matrix
    >>> C0 = np.dot(X.T, X)

    We compute the full correlation matrix as a reference and start oASIS with 3 out of 10 columns:

    >>> # approximate correlation matrix
    >>> d = np.diag(C0)
    >>> cols = np.array([0, 4, 9])
    >>> C0_k = C0[:, cols]
    >>> oasis = oASIS_Nystroem(d, C0_k, cols)
    >>> # show error of the current approximation
    >>> print('{:.2e}'.format(np.max(np.abs(oasis.error))))
    1.00e+04

    Now we conduct the approximation. We ask oASIS which columns should be computed next, compute them with whichever
    algorithm applies, and update the oASIS approximation. This can be repeated until the error is small enough or
    until a certain number of columns is reached.

    >>> # ask oASIS which column we should compute next
    >>> newcol = oasis.select_columns()
    >>> # recompute the new column yourself
    >>> c = np.dot(X.T, X[:, newcol])
    >>> # update oASIS
    >>> oasis.add_columns(c, newcol)
    array([1, 2, 3])
    >>> # take note of the new column index
    >>> cols = np.append(cols, newcol)
    >>> # show error of the current approximation
    >>> np.max(np.abs(oasis.error)) < 1e-10
    True

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
        r"""
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
        # err_i = sum_j R_{k,ij} A_{k,ji} - d_i
        self._err = np.sum(np.multiply(self._R_k, self._C_k.T), axis=0) - self._d

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
        return np.dot(self._C_k, self._R_k)

    def approximate_column(self, i):
        r""" Computes the Nystroem approximation of column :math:`i` of matrix $A \in \mathbb{R}^{n \times n}$.

        """
        return np.dot(self._C_k, self._R_k[:, i])

    def approximate_cholesky(self, epsilon=1e-6):
        r""" Compute low-rank approximation to the Cholesky decomposition of target matrix.

        The decomposition will be conducted while ensuring that the spectrum of `A_k^{-1}` is positive.

        Parameters
        ----------
        epsilon : float, optional, default 1e-6
            Cutoff for eigenvalue norms. If negative eigenvalues occur, with norms larger than epsilon, the largest
            negative eigenvalue norm will be used instead of epsilon, i.e. a band including all negative eigenvalues
            will be cut off.

        Returns
        -------
        L : ndarray((n,m), dtype=float)
            Cholesky matrix such that `A \approx L L^{\top}`. Number of columns :math:`m` is most at the number of columns
            used in the Nystroem approximation, but may be smaller depending on epsilon.

        """
        # compute the Eigenvalues of C0 using Schur factorization
        Wk = self._C_k[self._columns, :]
        L0 = spd_inv_split(Wk, epsilon=epsilon)
        L = np.dot(self._C_k, L0)

        return L

    def approximate_eig(self, epsilon=1e-6):
        """ Compute low-rank approximation of the eigenvalue decomposition of target matrix.

        If spd is True, the decomposition will be conducted while ensuring that the spectrum of `A_k^{-1}` is positive.

        Parameters
        ----------
        epsilon : float, optional, default 1e-6
            Cutoff for eigenvalue norms. If negative eigenvalues occur, with norms larger than epsilon, the largest
            negative eigenvalue norm will be used instead of epsilon, i.e. a band including all negative eigenvalues
            will be cut off.

        Returns
        -------
        s : ndarray((m,), dtype=float)
            approximated eigenvalues. Number of eigenvalues returned is at most the number of columns used in the
            Nystroem approximation, but may be smaller depending on epsilon.

        W : ndarray((n,m), dtype=float)
            approximated eigenvectors in columns. Number of eigenvectors returned is at most the number of columns
            used in the Nystroem approximation, but may be smaller depending on epsilon.

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


class SelectionStrategy(object):
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
        if self._oasis_obj._n == self._oasis_obj._k:  # nothing left to select?
            warnings.warn('Requested more columns but there are none left. Returning None.')
            return None
        if self._nsel > self._oasis_obj._n - self._oasis_obj._k:  # less columns left than requested
            ncols = self._oasis_obj._n - self._oasis_obj._k
            warnings.warn('Requested more columns than are left to select. Returning only '+str(ncols)+' columns.')
            return ncols
        return self._nsel

    def select(self):
        """ Selects next column indexes according to defined strategy

        Returns
        -------
        cols : ndarray((nsel,), dtype=int)
            selected columns

        """
        err = self._oasis_obj.error
        if np.allclose(err, 0):
            return None
        nsel = self._check_nsel()
        if nsel is None:
            return None
        return self._select(nsel, err)

    def _select(self, nsel, err):
        raise NotImplementedError('Classes derived from SelectionStrategy must override the _select() method.')


class SelectionStrategyRandom(SelectionStrategy):
    """ Selects nsel random columns not yet included in the approximation """
    def _select(self, nsel, err):
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
        if np.allclose(self._oasis_obj.Wk, 0):
            evec = np.ones((self._oasis_obj.k, self._oasis_obj.k))
        else:
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
        evec0 = evecs[:, 0]
        if np.isclose(np.min(evec0), np.max(evec0)):
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
