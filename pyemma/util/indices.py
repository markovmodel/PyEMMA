
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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


def combinations(seq, k):
    """ Return j length subsequences of elements from the input iterable.

    This version uses Numpy/Scipy and should be preferred over itertools. It avoids
    the creation of all intermediate Python objects.

    Examples
    --------

    >>> import numpy as np
    >>> from itertools import combinations as iter_comb
    >>> x = np.arange(3)
    >>> c1 = combinations(x, 2)
    >>> print(c1)
       [[0 1]
       [0 2]
       [1 2]]
    >>> c2 = np.array(tuple(iter_comb(x, 2)))
    >>> print(c2)
       [[0 1]
       [0 2]
       [1 2]]
    """
    from itertools import combinations as _combinations, chain
    from scipy.misc import comb

    count = comb(len(seq), k, exact=True)
    res = np.fromiter(chain.from_iterable(_combinations(seq, k)),
                      int, count=count*k)
    return res.reshape(-1, k)


def product(*arrays):
    """ Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)
    dtype = arrays[0].dtype

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    out = np.empty_like(ix, dtype=dtype)

    for n, _ in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out
