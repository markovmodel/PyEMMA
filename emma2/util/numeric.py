'''
Created on 28.10.2013

@author: marscher
'''
import numpy as np
from scipy.sparse import dia_matrix

__all__ = ['allclose_sparse', 'diags']

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
