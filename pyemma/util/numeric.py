'''
Created on 28.10.2013

@author: marscher
'''
import numpy as np

from scipy.sparse import dia_matrix

__all__ = ['allclose_sparse',
           'assert_allclose',
           'diags',
           'isclose',
           'choice',
           ]

def allclose_sparse(A, B, rtol=1e-5, atol=1e-9):
    """
    Compares two sparse matrices in the same matter like numpy.allclose()
    Parameters
    ----------
    A : scipy.sparse matrix
        first matrix to compare
    B : scipy.sparse matrix
        second matrix to compare
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance
    
    Returns
    -------
    True, if given matrices are equal in bounds of rtol and atol
    False, otherwise
    
    Notes
    -----
    If the following equation is element-wise True, then allclose returns
    True.

     absolute(`a` - `b`) <= (`atol` + `rtol` * absolute(`b`))

    The above equation is not symmetric in `a` and `b`, so that
    `allclose(a, b)` might be different from `allclose(b, a)` in
    some rare cases.
    """
    diff = (A - B).data
    return np.allclose(diff, 0.0, rtol=rtol, atol=atol)

################################################################################
# Backward compatibility to NumPy 1.6 and scipy 0.11
################################################################################
# Some functions which are being used from future versions are copied here and
# are only used, if we are not able to import them.
################################################################################

try:
    from scipy.sparse import diags
except ImportError:
    def diags(diagonals, offsets, shape=None, format=None, dtype=None):
        """
Note: copied from scipy.sparse.construct

Construct a sparse matrix from diagonals.

.. versionadded:: 0.11

Parameters
----------
diagonals : sequence of array_like
Sequence of arrays containing the matrix diagonals,
corresponding to `offsets`.
offsets : sequence of int
Diagonals to set:
- k = 0 the main diagonal
- k > 0 the k-th upper diagonal
- k < 0 the k-th lower diagonal
shape : tuple of int, optional
Shape of the result. If omitted, a square matrix large enough
to contain the diagonals is returned.
format : {"dia", "csr", "csc", "lil", ...}, optional
Matrix format of the result. By default (format=None) an
appropriate sparse matrix format is returned. This choice is
subject to change.
dtype : dtype, optional
Data type of the matrix.

See Also
--------
spdiags : construct matrix from diagonals

Notes
-----
This function differs from `spdiags` in the way it handles
off-diagonals.

The result from `diags` is the sparse equivalent of::

np.diag(diagonals[0], offsets[0])
+ ...
+ np.diag(diagonals[k], offsets[k])

Repeated diagonal offsets are disallowed.

Examples
--------
>>> diagonals = [[1,2,3,4], [1,2,3], [1,2]]
>>> diags(diagonals, [0, -1, 2]).todense()
matrix([[1., 0., 1., 0.],
[1., 2., 0., 2.],
[0., 2., 3., 0.],
[0., 0., 3., 4.]])

Broadcasting of scalars is supported (but shape needs to be
specified):

>>> diags([1, -2, 1], [-1, 0, 1], shape=(4, 4)).todense()
matrix([[-2., 1., 0., 0.],
[ 1., -2., 1., 0.],
[ 0., 1., -2., 1.],
[ 0., 0., 1., -2.]])


If only one diagonal is wanted (as in `numpy.diag`), the following
works as well:

>>> diags([1, 2, 3], 1).todense()
matrix([[ 0., 1., 0., 0.],
[ 0., 0., 2., 0.],
[ 0., 0., 0., 3.],
[ 0., 0., 0., 0.]])
"""
        # if offsets is not a sequence, assume that there's only one diagonal
        try:
            iter(offsets)
        except TypeError:
            # now check that there's actually only one diagonal
            try:
                iter(diagonals[0])
            except TypeError:
                diagonals = [np.atleast_1d(diagonals)]
            else:
                raise ValueError("Different number of diagonals and offsets.")
        else:
            diagonals = list(map(np.atleast_1d, diagonals))
        offsets = np.atleast_1d(offsets)
    
        # Basic check
        if len(diagonals) != len(offsets):
            raise ValueError("Different number of diagonals and offsets.")
    
        # Determine shape, if omitted
        if shape is None:
            m = len(diagonals[0]) + abs(int(offsets[0]))
            shape = (m, m)
    
        # Determine data type, if omitted
        if dtype is None:
            dtype = np.common_type(*diagonals)
    
        # Construct data array
        m, n = shape
    
        M = max([min(m + offset, n - offset) + max(0, offset)
                 for offset in offsets])
        M = max(0, M)
        data_arr = np.zeros((len(offsets), M), dtype=dtype)
    
        for j, diagonal in enumerate(diagonals):
            offset = offsets[j]
            k = max(0, offset)
            length = min(m + offset, n - offset)
            if length <= 0:
                raise ValueError("Offset %d (index %d) out of bounds" % (offset, j))
            try:
                data_arr[j, k:k+length] = diagonal
            except ValueError:
                if len(diagonal) != length and len(diagonal) != 1:
                    raise ValueError(
                        "Diagonal length (index %d: %d at offset %d) does not "
                        "agree with matrix size (%d, %d)." % (
                        j, len(diagonal), offset, m, n))
                raise
    
        return dia_matrix((data_arr, offsets), shape=(m, n)).asformat(format)

try:
    from np import isclose
except ImportError:
    def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
        def within_tol(x, y, atol, rtol):
            result = np.less_equal(np.abs(x-y), atol + rtol * np.abs(y))
            if np.isscalar(a) and np.isscalar(b):
                result = np.bool(result)
            return result

        x = np.array(a, copy=False, subok=True, ndmin=1)
        y = np.array(b, copy=False, subok=True, ndmin=1)
        xfin = np.isfinite(x)
        yfin = np.isfinite(y)
        if np.all(xfin) and np.all(yfin):
            return within_tol(x, y, atol, rtol)
        else:
            finite = xfin & yfin
            cond = np.zeros_like(finite, subok=True)
            # Because we're using boolean indexing, x & y must be the same shape.
            # Ideally, we'd just do x, y = broadcast_arrays(x, y). It's in
            # lib.stride_tricks, though, so we can't import it here.
            x = x * np.ones_like(cond)
            y = y * np.ones_like(cond)
            # Avoid subtraction with infinite/nan values...
            cond[finite] = within_tol(x[finite], y[finite], atol, rtol)
            # Check for equality of infinite values...
            cond[~finite] = (x[~finite] == y[~finite])
            if equal_nan:
                # Make NaN == NaN
                cond[np.isnan(x) & np.isnan(y)] = True
            return cond

try:
    from np.random import choice
except ImportError:
    def choice(a, size=None, replace=True, p=None):
        """
        choice(a, size=1, replace=True, p=None)

        Generates a random sample from a given 1-D array

                .. versionadded:: 1.7.0

        Parameters
        -----------
        a : 1-D array-like or int
            If an ndarray, a random sample is generated from its elements.
            If an int, the random sample is generated as if a was np.arange(n)
        size : int or tuple of ints, optional
            Output shape. Default is None, in which case a single value is
            returned.
        replace : boolean, optional
            Whether the sample is with or without replacement
        p : 1-D array-like, optional
            The probabilities associated with each entry in a.
            If not given the sample assumes a uniform distribtion over all
            entries in a.

        Returns
        --------
        samples : 1-D ndarray, shape (size,)
            The generated random samples

        Raises
        -------
        ValueError
            If a is an int and less than zero, if a or p are not 1-dimensional,
            if a is an array-like of size 0, if p is not a vector of
            probabilities, if a and p have different lengths, or if
            replace=False and the sample size is greater than the population
            size

        See Also
        ---------
        randint, shuffle, permutation

        Examples
        ---------
        Generate a uniform random sample from np.arange(5) of size 3:

        >>> np.random.choice(5, 3)
        array([0, 3, 4])
        >>> #This is equivalent to np.random.randint(0,5,3)

        Generate a non-uniform random sample from np.arange(5) of size 3:

        >>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
        array([3, 3, 0])

        Generate a uniform random sample from np.arange(5) of size 3 without
        replacement:

        >>> np.random.choice(5, 3, replace=False)
        array([3,1,0])
        >>> #This is equivalent to np.random.shuffle(np.arange(5))[:3]

        Generate a non-uniform random sample from np.arange(5) of size
        3 without replacement:

        >>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
        array([2, 3, 0])

        Any of the above can be repeated with an arbitrary array-like
        instead of just integers. For instance:

        >>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
        >>> np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
        array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'],
              dtype='|S11')

        """

        # Format and Verify input
        a = np.array(a, copy=False)
        if a.ndim == 0:
            try:
                import operator
                if hasattr(operator, "index"): # python 2.5+
                    # __index__ must return an integer by python rules.
                    pop_size = operator.index(a.item())
                else:
                    pop_size = int(a.item())
            except (TypeError, ValueError):
                raise ValueError("a must be 1-dimensional or an integer")
            if pop_size <= 0:
                raise ValueError("a must be greater than 0")
        elif a.ndim != 1:
            raise ValueError("a must be 1-dimensional")
        else:
            pop_size = a.shape[0]
            if pop_size is 0:
                raise ValueError("a must be non-empty")

        if None != p:
            p = np.array(p, dtype=np.double, ndmin=1, copy=False)
            if p.ndim != 1:
                raise ValueError("p must be 1-dimensional")
            if p.size != pop_size:
                raise ValueError("a and p must have same size")
            if np.any(p < 0):
                raise ValueError("probabilities are not non-negative")
            if not np.allclose(p.sum(), 1):
                raise ValueError("probabilities do not sum to 1")

        shape = size
        if shape is not None:
            size = np.prod(shape, dtype=np.intp)
        else:
            size = 1

        # Actual sampling
        if replace:
            if None != p:
                cdf = p.cumsum()
                cdf /= cdf[-1]
                uniform_samples = np.random.random_sample(shape)
                idx = cdf.searchsorted(uniform_samples, side='right')
                idx = np.array(idx, copy=False) # searchsorted returns a scalar
            else:
                idx = np.random.randint(0, pop_size, size=shape)
        else:
            if size > pop_size:
                raise ValueError("Cannot take a larger sample than "
                                 "population when 'replace=False'")

            if None != p:
                if np.sum(p > 0) < size:
                    raise ValueError("Fewer non-zero entries in p than size")
                n_uniq = 0
                p = p.copy()
                found = np.zeros(shape, dtype=np.int)
                flat_found = found.ravel()
                while n_uniq < size:
                    x = np.random.rand(size - n_uniq)
                    if n_uniq > 0:
                        p[flat_found[0:n_uniq]] = 0
                    cdf = np.cumsum(p)
                    cdf /= cdf[-1]
                    new = cdf.searchsorted(x, side='right')
                    _, unique_indices = np.unique(new, return_index=True)
                    unique_indices.sort()
                    new = new.take(unique_indices)
                    flat_found[n_uniq:n_uniq + new.size] = new
                    n_uniq += new.size
                idx = found
            else:
                idx = np.random.permutation(pop_size)[:size]
                if shape is not None:
                    idx.shape = shape

        if shape is None and isinstance(idx, np.ndarray):
            # In most cases a scalar will have been made an array
            idx = idx.item(0)

        #Use samples as indices for a if a is array-like
        if a.ndim == 0:
            return idx

        if shape is not None and idx.ndim == 0:
            # If size == () then the user requested a 0-d array as opposed to
            # a scalar object when size is None. However a[idx] is always a
            # scalar and not an array. So this makes sure the result is an
            # array, taking into account that np.array(item) may not work
            # for object arrays.
            res = np.empty((), dtype=a.dtype)
            res[()] = a[idx]
            return res

        return a[idx]


def assert_allclose(actual, desired, rtol=1.e-5, atol=1.e-8,
                    err_msg='', verbose=True):
    r"""wrapper for numpy.testing.allclose with default tolerances of
    numpy.allclose. Needed since testing method has different values."""
    from numpy.testing import assert_allclose
    return assert_allclose(actual, desired, rtol, atol, err_msg, verbose)
