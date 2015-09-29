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
Python interface to the WHAM estimator's lowlevel functions.
"""

import numpy as _np
cimport numpy as _np

__all__ = ['iterate_fi', 'iterate_fk', 'normalize', 'estimate']

cdef extern from "_wham.h":
    void _iterate_fi(
        double *log_N_K, double *log_N_i, double *f_K, double *b_K_i,
        int n_therm_states, int n_markov_states, double *scratch_T, double *f_i)
    void _iterate_fk(
        double *f_i, double *b_K_i, int n_therm_states, int n_markov_states,
        double *scratch_M, double *f_K)
    void _normalize(
        double *f_K, double *f_i, int n_therm_states, int n_markov_states, double *scratch_M)

def iterate_fi(
    _np.ndarray[double, ndim=1, mode="c"] log_N_K not None,
    _np.ndarray[double, ndim=1, mode="c"] log_N_i not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] f_i not None):
    r"""
    Calculate the reduced free energies f_i
        
    Parameters
    ----------
    log_N_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    log_N_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        log of the state counts in each of the M markov states
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        bias energies in the T thermodynamic and M discrete Markov states
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states

    Notes
    -----
    The iterate_fi() function computes

    .. math:
        f_i = -\ln\left( \frac{
                \sum_{K=0}^{N_T-1} N_i^{(K)}
            }{
                \sum_{K=0}^{N_T-1} \exp(f^{(K)}-b_i^{(K)}) \sum_{j=0}^{N_M-1} N_j^{(K)}
            }\right)

    which equals to

    .. math:
        f_i = \ln\left(
                \sum_{K=0}^{N_T-1} \exp\left(
                    f^{(K)} - b_i^{(K)} + \ln\left( \sum_{j=0}^{N_M-1} N_j^{(K)} \right)
                \right)
            \right) - \ln\left( \sum_{K=0}^{N_T-1} N_i^{(K)} \right)

    in the logsumexp scheme. Afterwards, we apply

    .. math:
        f_i \leftarrow  f_i - \min_i(f_i)

    """
    _iterate_fi(
        <double*> _np.PyArray_DATA(log_N_K),
        <double*> _np.PyArray_DATA(log_N_i),
        <double*> _np.PyArray_DATA(f_K),
        <double*> _np.PyArray_DATA(b_K_i),
        b_K_i.shape[0],
        b_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(f_i))

def iterate_fk(
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] f_K not None):
    r"""
    Calculate the reduced thermodynamic free energies f_K
        
    Parameters
    ----------
    f_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced free energies of the M markov states
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        bias energies in the T thermodynamic and M discrete Markov states
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states

    Notes
    -----
    The iterate_fk() function computes

    .. math:
        f^{(K)} = -\ln\left(
                \sum_{i=0}^{N_M-1} \exp(-b_i^{(K)}-f_i)
            \right)

    which is already in the logsumexp form.
    """
    _iterate_fk(
        <double*> _np.PyArray_DATA(f_i),
        <double*> _np.PyArray_DATA(b_K_i),
        b_K_i.shape[0],
        b_K_i.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(f_K))

def normalize(
    _np.ndarray[double, ndim=1, mode="c"] f_K not None,
    _np.ndarray[double, ndim=1, mode="c"] f_i not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None):
    r"""
    Normalize the unbiased reduced free energies and shift the thermodynamic
    free energies accordingly
        
    Parameters
    ----------
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    f_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced free energies of the M discrete states
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    """
    _normalize(
        <double*> _np.PyArray_DATA(f_K),
        <double*> _np.PyArray_DATA(f_i),
        f_K.shape[0],
        f_i.shape[0],
        <double*> _np.PyArray_DATA(scratch_M))

def estimate(N_K_i, b_K_i, maxiter=1000, maxerr=1.0E-8, f_K=None, f_i=None):
    r"""
    Estimate the unbiased reduced free energies and thermodynamic free energies
        
    Parameters
    ----------
    N_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        discrete state counts in the T thermodynamic and M discrete states
    b_K_i : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic and M discrete states
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced free energies of the T thermodynamic states
    f_i : numpy.ndarray(shape=(M), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced unbiased free energies of the M discrete states

    Returns
    -------
    f_K : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    f_i : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased free energies of the M discrete states
    """
    T = N_K_i.shape[0]
    M = N_K_i.shape[1]
    S = _np.max([T, M])
    log_N_K = _np.log(N_K_i.sum(axis=1))
    log_N_i = _np.log(N_K_i.sum(axis=0))
    if f_K is None:
        f_K = _np.zeros(shape=(T,), dtype=_np.float64)
    if f_i is None:
        f_i = _np.zeros(shape=(M,), dtype=_np.float64)
    old_f_K = f_K.copy()
    old_f_i = f_i.copy()
    scratch = _np.zeros(shape=(S,), dtype=_np.float64)
    stop = False
    for _m in range(maxiter):
        iterate_fk(f_i, b_K_i, scratch, f_K)
        iterate_fi(log_N_K, log_N_i, f_K, b_K_i, scratch, f_i)
        delta_f_K = _np.max(_np.abs((f_K - old_f_K)))
        delta_f_i = _np.max(_np.abs((f_i - old_f_i)))
        if delta_f_K < maxerr and delta_f_i < maxerr:
            stop = True
        else:
            old_f_K[:] = f_K[:]
            old_f_i[:] = f_i[:]
        if stop:
            break
    normalize(f_K, f_i, scratch)
    return f_K, f_i
