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

from .callback import CallbackInterrupt

__all__ = [
    'update_conf_energies',
    'update_therm_energies',
    'normalize',
    'get_loglikelihood',
    'estimate']

cdef extern from "_wham.h":
    void _wham_update_conf_energies(
        double *log_therm_state_counts, double *log_conf_state_counts,
        double *therm_energies, double *bias_energies,
        int n_therm_states, int n_conf_states, double *scratch_T, double *conf_energies)
    void _wham_update_therm_energies(
        double *conf_energies, double *bias_energies, int n_therm_states, int n_conf_states,
        double *scratch_M, double *therm_energies)
    void _wham_normalize(
        int n_therm_states, int n_conf_states,
        double *scratch_M, double *therm_energies, double *conf_energies)
    double _wham_get_loglikelihood(
        int *therm_state_counts, int *conf_state_counts,
        double *therm_energies, double *conf_energies,
        int n_therm_states, int n_conf_states, double *scratch_S)

def update_conf_energies(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] log_conf_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None):
    r"""
    Calculate the reduced unbiased free energies.
        
    Parameters
    ----------
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        logarithm of the state counts in each of the T thermodynamic states
    log_conf_state_counts : numpy.ndarray(shape=(M), dtype=numpy.float64)
        logarithm of the state counts in each of the M configurational states
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M configurational states
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased configurational energies

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
    _wham_update_conf_energies(
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
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased configurational energies
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M configurational states
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
    _wham_update_therm_energies(
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
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased configurational energies
    """
    _wham_normalize(
        therm_energies.shape[0],
        conf_energies.shape[0],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(conf_energies))

def get_loglikelihood(
    _np.ndarray[int, ndim=1, mode="c"] therm_state_counts not None,
    _np.ndarray[int, ndim=1, mode="c"] conf_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_S not None):
    r"""
    Compute the loglikelihood of the estimated reduced free energies.

    Parameters
    ----------


    Returns
    -------
    loglikelihood : float
        loglikelihood of the reduced free energies given the observed state counts
    """
    return _wham_get_loglikelihood(
        <int*> _np.PyArray_DATA(therm_state_counts),
        <int*> _np.PyArray_DATA(conf_state_counts),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(conf_energies),
        therm_state_counts.shape[0],
        conf_state_counts.shape[0],
        <double*> _np.PyArray_DATA(scratch_S))

def estimate(
    state_counts, bias_energies,
    maxiter=1000, maxerr=1.0E-8,
    therm_energies=None, conf_energies=None,
    err_out=0, lll_out=0, callback=None):
    r"""
    Estimate the unbiased reduced free energies and thermodynamic free energies
        
    Parameters
    ----------
    state_counts : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        state counts in the T thermodynamic and M configurational states
    bias_energies : numpy.ndarray(shape=(T, M), dtype=numpy.intc)
        reduced bias energies of the T thermodynamic and M configurational states
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    therm_energies : numpy.ndarray(shape=(T,), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced thermodynamic energies
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        initial guess for the reduced unbiased free energies
    err_out : int, optional
        every err_out iteration steps, store the actual increment
    lll_out : int, optional
        every lll_out iteration steps, store the actual loglikelihood

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T,), dtype=numpy.float64)
        reduced thermodynamic energies
    conf_energies : numpy.ndarray(shape=(M,), dtype=numpy.float64)
        reduced unbiased configurational energies
    err : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of increments
    lll : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of loglikelihoods

    Notes
    -----
    This function calls the previously defined update functions to estimate the reduced
    configuration energies of the unbiased thermodynamic state and the reduced thermodynamic
    energies by means of a fixed point iteration.
    """
    T = state_counts.shape[0]
    M = state_counts.shape[1]
    S = T + M
    therm_state_counts = state_counts.sum(axis=1).astype(_np.intc)
    conf_state_counts = state_counts.sum(axis=0).astype(_np.intc)
    log_therm_state_counts = _np.log(therm_state_counts).astype(_np.float64)
    log_conf_state_counts = _np.log(conf_state_counts).astype(_np.float64)
    if therm_energies is None:
        therm_energies = _np.zeros(shape=(T,), dtype=_np.float64)
    if conf_energies is None:
        conf_energies = _np.zeros(shape=(M,), dtype=_np.float64)
    old_therm_energies = therm_energies.copy()
    old_conf_energies = conf_energies.copy()
    scratch = _np.zeros(shape=(S,), dtype=_np.float64)
    err_traj = []
    lll_traj = []
    err_count = 0
    lll_count = 0
    for m in range(maxiter):
        err_count += 1
        lll_count += 1
        update_therm_energies(conf_energies, bias_energies, scratch, therm_energies)
        update_conf_energies(
            log_therm_state_counts, log_conf_state_counts, therm_energies, bias_energies,
            scratch, conf_energies)
        delta_therm_energies = _np.abs(therm_energies - old_therm_energies)
        delta_conf_energies = _np.abs(conf_energies - old_conf_energies)
        err = _np.max([_np.max(delta_conf_energies), _np.max(delta_therm_energies)])
        normalize(scratch, therm_energies, conf_energies)
        if err_count == err_out:
            err_count = 0
            err_traj.append(err)
        if lll_count == lll_out:
            lll_count = 0
            lll_traj.append(
                get_loglikelihood(
                    therm_state_counts, conf_state_counts, therm_energies, conf_energies, scratch))
        if callback is not None:
            try:
                callback(
                    conf_energies=conf_energies,
                    old_therm_energies=old_therm_energies,
                    old_conf_energies=old_conf_energies,
                    therm_energies=therm_energies,
                    delta_conf_energies=delta_conf_energies,
                    delta_therm_energies=delta_therm_energies,
                    err=err,
                    iteration_step=m,
                    maxiter=maxiter,
                    maxerr=maxerr)
            except CallbackInterrupt:
                break
        if err < maxerr:
            break
        else:
            old_therm_energies[:] = therm_energies[:]
            old_conf_energies[:] = conf_energies[:]
    if err_out == 0:
        err_traj = None
    else:
        err_traj = _np.array(err_traj, dtype=_np.float64)
    if lll_out == 0:
        lll_traj = None
    else:
        lll_traj = _np.array(lll_traj, dtype=_np.float64)
    return therm_energies, conf_energies, err_traj, lll_traj
