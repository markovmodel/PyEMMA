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

__all__ = ['update_conf_energies', 'update_therm_energies', 'normalize', 'estimate']

cdef extern from "_wham.h":
    void _update_conf_energies(
        double *log_therm_state_counts, double *log_conf_state_counts, double *therm_energies, double *bias_energies,
        int n_therm_states, int n_conf_states, double *scratch_T, double *conf_energies)
    void _update_therm_energies(
        double *conf_energies, double *bias_energies, int n_therm_states, int n_conf_states,
        double *scratch_M, double *therm_energies)
    void _normalize(
        int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies, double *conf_energies)

def update_conf_energies(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] log_conf_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None):
    r"""
    Calculate the reduced free energies conf_energies
        
    Parameters
    ----------
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    log_conf_state_counts : numpy.ndarray(shape=(M), dtype=numpy.float64)
        log of the state counts in each of the M configurational states
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        bias energies in the T thermodynamic and M discrete configurational states
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states

    Notes
    -----
    The update_conf_energies() function computes

    .. math:
        conf_energies = -\ln\left( \frac{
                \sum_{K=0}^{N_T-1} N_i^{(K)}
            }{
                \sum_{K=0}^{N_T-1} \exp(f^{(K)}-b_i^{(K)}) \sum_{j=0}^{N_M-1} N_j^{(K)}
            }\right)

    which equals to

    .. math:
        conf_energies = \ln\left(
                \sum_{K=0}^{N_T-1} \exp\left(
                    f^{(K)} - b_i^{(K)} + \ln\left( \sum_{j=0}^{N_M-1} N_j^{(K)} \right)
                \right)
            \right) - \ln\left( \sum_{K=0}^{N_T-1} N_i^{(K)} \right)

    in the logsumexp scheme. Afterwards, we apply

    .. math:
        conf_energies \leftarrow  conf_energies - \min_i(conf_energies)

    """
    _update_conf_energies(
        <double*> _np.PyArray_DATA(log_therm_state_counts),
        <double*> _np.PyArray_DATA(log_conf_state_counts),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(bias_energies),
        bias_energies.shape[0],
        bias_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(conf_energies))

def update_therm_energies(
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None):
    r"""
    Calculate the reduced thermodynamic free energies therm_energies
        
    Parameters
    ----------
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced free energies of the M configurational states
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        bias energies in the T thermodynamic and M configurational states
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states

    Notes
    -----
    The update_therm_energies() function computes

    .. math:
        f^{(K)} = -\ln\left(
                \sum_{i=0}^{N_M-1} \exp(-b_i^{(K)}-conf_energies)
            \right)

    which is already in the logsumexp form.
    """
    _update_therm_energies(
        <double*> _np.PyArray_DATA(conf_energies),
        <double*> _np.PyArray_DATA(bias_energies),
        bias_energies.shape[0],
        bias_energies.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(therm_energies))

def normalize(
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None):
    r"""
    Normalize the unbiased reduced free energies and shift the thermodynamic
    free energies accordingly
        
    Parameters
    ----------
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced free energies of the M discrete states
    """
    _normalize(
        therm_energies.shape[0],
        conf_energies.shape[0],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(conf_energies))

def estimate(state_counts, bias_energies, maxiter=1000, maxerr=1.0E-8, therm_energies=None, conf_energies=None):
    r"""
    Estimate the unbiased reduced free energies and thermodynamic free energies
        
    Parameters
    ----------
    state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        state counts in the T thermodynamic and M configurational states
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic and M configurational states
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced free energies of the T thermodynamic states
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced unbiased free energies of the M configurational states

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased free energies of the M configurational states
    """
    T = state_counts.shape[0]
    M = state_counts.shape[1]
    S = _np.max([T, M])
    log_therm_state_counts = _np.log(state_counts.sum(axis=1))
    log_conf_state_counts = _np.log(state_counts.sum(axis=0))
    if therm_energies is None:
        therm_energies = _np.zeros(shape=(T,), dtype=_np.float64)
    if conf_energies is None:
        conf_energies = _np.zeros(shape=(M,), dtype=_np.float64)
    old_therm_energies = therm_energies.copy()
    old_conf_energies = conf_energies.copy()
    scratch = _np.zeros(shape=(S,), dtype=_np.float64)
    stop = False
    for _m in range(maxiter):
        update_therm_energies(conf_energies, bias_energies, scratch, therm_energies)
        update_conf_energies(log_therm_state_counts, log_conf_state_counts, therm_energies, bias_energies, scratch, conf_energies)
        delta_therm_energies = _np.max(_np.abs((therm_energies - old_therm_energies)))
        delta_conf_energies = _np.max(_np.abs((conf_energies - old_conf_energies)))
        if delta_therm_energies < maxerr and delta_conf_energies < maxerr:
            stop = True
        else:
            old_therm_energies[:] = therm_energies[:]
            old_conf_energies[:] = conf_energies[:]
        if stop:
            break
    normalize(scratch, therm_energies, conf_energies)
    return therm_energies, conf_energies
