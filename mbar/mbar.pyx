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
Python interface to the MBAR estimator's lowlevel functions.
"""

import numpy as _np
cimport numpy as _np
from .callback import CallbackInterrupt

__all__ = ['update_therm_energies', 'normalize', 'get_conf_energies', 'get_biased_conf_energies', 'estimate']

cdef extern from "_mbar.h":
    void _mbar_update_therm_energies(
        double *log_therm_state_counts, double *therm_energies, double *bias_energy_sequence,
        int n_therm_states, int seq_length, double *scratch_T, double *new_therm_energies)
    void _mbar_get_conf_energies(
        double *log_therm_state_counts, double *therm_energies,
        double *bias_energy_sequence, int * conf_state_sequence,
        int n_therm_states, int n_conf_states, int seq_length,
        double *scratch_T, double *conf_energies, double *biased_conf_energies)
    void _mbar_normalize(
        double *log_therm_state_counts, double *bias_energy_sequence,
        int n_therm_states, int n_conf_states, int seq_length, double *scratch_M,
        double *therm_energies, double *conf_energies, double *biased_conf_energies)

def update_therm_energies(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] new_therm_energies not None):
    r"""
    Calculate the reduced thermodynamic free energies therm_energies
        
    Parameters
    ----------
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    new_therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states
    """
    _mbar_update_therm_energies(
        <double*> _np.PyArray_DATA(log_therm_state_counts),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        bias_energy_sequence.shape[0],
        bias_energy_sequence.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(new_therm_energies))

def get_conf_energies(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[int, ndim=1, mode="c"] conf_state_sequence not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    n_conf_states):
    r"""
    Calculate the reduced unbiased free energies conf_energies
        
    Parameters
    ----------
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    conf_state_sequence : numpy.ndarray(shape=(X), dtype=numpy.intc)
        discrete states indices for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    n_conf_states : int
        number of discrete states (M)

    Returns
    -------
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased free energies
    """
    conf_energies = _np.zeros(shape=(n_conf_states,), dtype=_np.float64)
    biased_conf_energies = _np.zeros(shape=(therm_energies.shape[0], n_conf_states), dtype=_np.float64)
    _mbar_get_conf_energies(
        <double*> _np.PyArray_DATA(log_therm_state_counts),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        <int*> _np.PyArray_DATA(conf_state_sequence),
        bias_energy_sequence.shape[0],
        n_conf_states,
        bias_energy_sequence.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(conf_energies),
        <double*> _np.PyArray_DATA(biased_conf_energies))
    return conf_energies, biased_conf_energies

def normalize(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None):
    r"""
    Shift the reduced thermodynamic free energies therm_energies such that the unbiased thermodynamic
    free energy is zero
        
    Parameters
    ----------
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
        scratch array
    """
    _mbar_normalize(
        <double*> _np.PyArray_DATA(log_therm_state_counts),
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        therm_energies.shape[0],
        conf_energies.shape[0],
        bias_energy_sequence.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(conf_energies),
        <double*> _np.PyArray_DATA(biased_conf_energies))

def estimate(therm_state_counts, bias_energy_sequence, conf_state_sequence,
    maxiter=1000, maxerr=1.0E-8, therm_energies=None, err_out=0, callback=None, n_conf_states=None):
    r"""
    Estimate the (un)biased reduced free energies and thermodynamic free energies
        
    Parameters
    ----------
    therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.intc)
        numbers of samples in the T thermodynamic states
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic states for all X samples
    conf_state_sequence : numpy.ndarray(shape=(X), dtype=numpy.float64)
        discrete state indices (cluster indices) for all X samples
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced free energies of the T thermodynamic states
    err_out : int, optional
        every err_out iteration steps, store the actual increment
    n_conf_states : int, optional, default=None
        the number of configurational states in `conf_state_sequence`.
        If None, this is set to max(conf_state_sequence)+1.

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased discrete state (cluster) free energies
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced discrete state free energies for all combinations of
        T thermodynamic states and M discrete states
    err : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of increments
    """
    T = therm_state_counts.shape[0]
    if n_conf_states is None:
        M = 1 + _np.max(conf_state_sequence)
    else:
        M = n_conf_states
    log_therm_state_counts = _np.log(therm_state_counts)
    if therm_energies is None:
        therm_energies = _np.zeros(shape=(T,), dtype=_np.float64)
    old_therm_energies = therm_energies.copy()
    err_traj = []
    err_count = 0
    scratch_T = _np.zeros(shape=(T,), dtype=_np.float64)
    scratch_M = _np.zeros(shape=(M,), dtype=_np.float64)
    stop = False
    for _m in range(maxiter):
        err_count += 1
        update_therm_energies(log_therm_state_counts, old_therm_energies, bias_energy_sequence, scratch_T, therm_energies)
        delta_therm_energies = _np.abs(therm_energies - old_therm_energies)
        err = _np.max(delta_therm_energies)
        if err_count == err_out:
            err_count = 0
            err_traj.append(err)
        if callback is not None:
            try:
                callback(therm_energies=therm_energies,
                         old_therm_energies=old_therm_energies,
                         delta_therm_energies=delta_therm_energies,
                         iteration_step=_m,
                         err=err,
                         maxerr=maxerr,
                         maxiter=maxiter)
            except CallbackInterrupt:
                stop = True
        if err < maxerr:
            stop = True
        else:
            old_therm_energies[:] = therm_energies[:]
        if stop:
            break
    conf_energies, biased_conf_energies = get_conf_energies(
        log_therm_state_counts, therm_energies, bias_energy_sequence, conf_state_sequence, scratch_T, M)
    normalize(log_therm_state_counts, bias_energy_sequence, scratch_M, therm_energies, conf_energies, biased_conf_energies)
    if err_out == 0:
        err_traj = None
    else:
        err_traj = _np.array(err_traj, dtype=_np.float64)
    return therm_energies, conf_energies, biased_conf_energies, err_traj
