# This file is part of thermotools.
#
# Copyright 2015 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# thermotools is free software: you can redistribute it and/or modify
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

r"""
Python interface to utility functions
"""

import numpy as _np
cimport numpy as _np
from scipy.sparse import csr_matrix as _csr
from msmtools.estimation import count_matrix as _cm

__all__ = ['count_matrices', 'state_counts']

cdef extern from "_util.h":
    int _get_therm_state_break_points(int *T_x, int seq_length, int *break_points)

def get_therm_state_break_points(
    _np.ndarray[int, ndim=1, mode="c"] T_x not None):
    r"""
    Find thermodynamic state changes within a trajectory

    Parameters
    ----------
    T_x : numpy.ndarray(shape=(X), dtype=numpy.intc)
        Thermodynamic state sequence of a trajectory of length X

    Returns
    -------
    T_B : numpy.ndarray(shape=(B), dtype=numpy.intc)
        Sequence of first subsequence starting frames
    """
    T_B = _np.zeros(shape=(T_x.shape[0],), dtype=_np.intc)
    nb = _get_therm_state_break_points(
        <int*> _np.PyArray_DATA(T_x),
        T_x.shape[0],
        <int*> _np.PyArray_DATA(T_B))
    return _np.ascontiguousarray(T_B[:nb])

def count_matrices(dtraj, lag, sliding=True, sparse_return=True, ntherm=None, nstates=None):
    r"""
    Count transitions at given lagtime

    Parameters
    ----------
    dtraj : list of numpy.ndarray(shape=(X, 2), dtype=np.intc)
        list of discretized trajectories
    lag : int
        lagtime in trajectory steps
    sliding : bool, optional
        if true the sliding window approach is used for transition counting
    sparse_return : bool (optional)
        whether to return a 3D dense matrix or a list 2D sparse matrices
    ntherm : int, optional
        enforce ntherm count-matrices with shape=(nstates, nstates)
    nstates : int, optional
        enforce count-matrices with shape=(nstates, nstates) for all thermodynamic states

    Returns
    -------
    C_K : [scipy.sparse.coo_matrix] or numpy.ndarray(shape=(T, M, M))
        count matrices at given lag in coordinate list format
    """
    kmax = _np.max([d[:, 0].max() for d in dtraj])
    nmax = _np.max([d[:, 1].max() for d in dtraj])
    if ntherm is None:
        ntherm = kmax + 1
    elif ntherm < kmax + 1:
        raise ValueError("ntherm is smaller than the number of observed thermodynamic states")
    if nstates is None:
        nstates = nmax + 1
    elif nstates < nmax + 1:
        raise ValueError("nstates is smaller than the number of observed microstates")
    C_K = [_csr((nstates, nstates), dtype=_np.intc)] * ntherm
    for d in dtraj:
        bp = get_therm_state_break_points(_np.ascontiguousarray(d[:, 0]))
        for b in range(1, bp.shape[0]):
            C_K[d[bp[b - 1], 0]] += _cm(
                _np.ascontiguousarray(d[bp[b - 1]:bp[b], 1]), lag,
                sliding=sliding, sparse_return=True, nstates=nstates)
        C_K[d[bp[-1], 0]] += _cm(
            _np.ascontiguousarray(d[bp[-1]:, 1]), lag,
            sliding=sliding, sparse_return=True, nstates=nstates)
    if sparse_return:
        return C_K
    return _np.array([C.todense() for C in C_K], dtype=_np.intc)

def state_counts(dtraj, nstates=None, ntherm=None):
    r"""
    Count discrete states in all thermodynamic states

    Parameters
    ----------
    dtraj : list of numpy.ndarray(shape=(X, 2), dtype=np.intc)
        list of discretized trajectories
    nstates : int, optional
        enforce state count matrix with shape=(ntherm, nstates)
    ntherm : int, optional
        enforce state count matrix with shape=(ntherm, nstates)

    Returns
    -------
    N : numpy.ndarray(shape=(T, M))
        state counts
    """
    kmax = _np.max([d[:, 0].max() for d in dtraj])
    nmax = _np.max([d[:, 1].max() for d in dtraj])
    if ntherm is None:
        ntherm = kmax + 1
    elif ntherm < kmax + 1:
        raise ValueError("ntherm is smaller than the number of observed thermodynamic states")
    if nstates is None:
        nstates = nmax + 1
    elif nstates < nmax + 1:
        raise ValueError("nstates is smaller than the number of observed microstates")
    N = _np.zeros(shape=(ntherm, nstates), dtype=_np.intc)
    for d in dtraj:
        for K in range(ntherm):
            for i in range(nstates):
                N[K, i] += ((d[:, 0] == K) * (d[:, 1] == i)).sum()
    return N
