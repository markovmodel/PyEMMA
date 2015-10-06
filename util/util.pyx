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

def count_matrices(dtraj, lag, sliding=True, sparse_return=True, nthermo=None, nstates=None):
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
    nthermo : int, optional
        enforce nthermo count-matrices with shape=(nstates, nstates)
    nstates : int, optional
        enforce count-matrices with shape=(nstates, nstates) for all thermodynamic states

    Returns
    -------
    C_K : [scipy.sparse.coo_matrix] or numpy.ndarray(shape=(T, M, M))
        count matrices at given lag in coordinate list format
    """
    kmax = _np.max([d[:, 0].max() for d in dtraj])
    nmax = _np.max([d[:, 1].max() for d in dtraj])
    if nthermo is None:
        nthermo = kmax + 1
    elif nthermo < kmax + 1:
        raise ValueError("nthermo is smaller than the number of observed thermodynamic states")
    if nstates is None:
        nstates = nmax + 1
    elif nstates < nmax + 1:
        raise ValueError("nstates is smaller than the number of observed microstates")
    C_K = [_csr((nstates, nstates), dtype=_np.intc)] * nthermo
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

def state_counts(dtraj, nstates=None, nthermo=None):
    r"""
    Count discrete states in all thermodynamic states

    Parameters
    ----------
    dtraj : list of numpy.ndarray(shape=(X, 2), dtype=np.intc)
        list of discretized trajectories
    nstates : int, optional
        enforce state count matrix with shape=(nthermo, nstates)
    nthermo : int, optional
        enforce state count matrix with shape=(nthermo, nstates)

    Returns
    -------
    N : numpy.ndarray(shape=(T, M))
        state counts
    """
    kmax = int(_np.max([d[:, 0].max() for d in dtraj]))
    nmax = int(_np.max([d[:, 1].max() for d in dtraj]))
    if nthermo is None:
        nthermo = kmax + 1
    elif nthermo < kmax + 1:
        raise ValueError("nthermo is smaller than the number of observed thermodynamic states")
    if nstates is None:
        nstates = nmax + 1
    elif nstates < nmax + 1:
        raise ValueError("nstates is smaller than the number of observed microstates")
    N = _np.zeros(shape=(nthermo, nstates), dtype=_np.intc)
    for d in dtraj:
        for K in range(nthermo):
            for i in range(nstates):
                N[K, i] += ((d[:, 0] == K) * (d[:, 1] == i)).sum()
    return N

def restrict_samples_to_cset(state_sequence, bias_energy_sequence, cset):
    r"""
    Restrict full list of samples to a subset and relabel configurational state indices

    Parameters
    ----------
    state_sequence : numpy.ndarray(shape=(X, 2), dtype=numpy.intc)
        sequence of the thermodynamic and configurational state indices of the X samples
    bias_energy_sequence : numpy.ndarray(shape=(X, T), dtype=numpy.float64)
        sequence of the reduced bias energies for all X samples in all T thermodynamic states
    cset : list
        list of configurational states within the desired set

    Returns
    -------
    new_state_sequence : numpy.ndarray(shape=(Y, 2), dtype=numpy.intc)
        restricted and relabeled sequence of the thermodynamic and configurational state
        indices of the Y valid samples
    new_bias_energy_sequence : numpy.ndarray(shape=(Y, T), dtype=numpy.float64)
        restricted sequence of the reduced bias energies for all Y valid samples in
        all T thermodynamic states
    """
    nmax = int(_np.max([_np.max(state_sequence), _np.max(cset)]))
    mapping = []
    o = 0
    for i in range(nmax + 1):
        if i in cset:
            mapping.append(o)
            o += 1
        else:
            mapping.append(-1)
    mapping = _np.array(mapping, dtype=_np.intc)
    conf_state_sequence = mapping[state_sequence[:, 1]]
    valid_samples = (conf_state_sequence != -1)
    new_state_sequence = _np.ascontiguousarray(state_sequence[valid_samples, :])
    new_state_sequence[:, 1] = conf_state_sequence[valid_samples]
    new_bias_energy_sequence = _np.ascontiguousarray(bias_energy_sequence[valid_samples, :])
    return new_state_sequence, new_bias_energy_sequence
