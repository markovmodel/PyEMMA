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

__all__ = ['update_therm_energies', 'normalize', 'get_conf_energies', 'estimate']

cdef extern from "_mbar.h":
    void _update_therm_energies(
        double *log_therm_state_counts, double *therm_energies, double *bias_energy_sequence,
        int n_therm_states, int seq_length, double *scratch_T, double *new_therm_energies)
    void _normalize(
        double *log_therm_state_counts, double *bias_energy_sequence,
        int n_therm_states, int seq_length,
        double *scratch_T, double *therm_energies)
    void _get_conf_energies(
        double *log_therm_state_counts, double *therm_energies,
        double *bias_energy_sequence, int * conf_state_sequence,
        int n_therm_states, int n_conf_states, int seq_length,
        double *scratch_M, double *scratch_T, double *conf_energies)

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
    _update_therm_energies(
        <double*> _np.PyArray_DATA(log_therm_state_counts),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        bias_energy_sequence.shape[0],
        bias_energy_sequence.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(new_therm_energies))

def normalize(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None):
    r"""
    Shift the reduced thermodynamic free energies therm_energies such that the unbiased thermodynamic
    free energy is zero
        
    Parameters
    ----------
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
        scratch array
    """
    _normalize(
        <double*> _np.PyArray_DATA(log_therm_state_counts),
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        bias_energy_sequence.shape[0],
        bias_energy_sequence.shape[1],
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(therm_energies))

def get_conf_energies(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] bias_energy_sequence not None,
    _np.ndarray[int, ndim=1, mode="c"] conf_state_sequence not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    n_discrete_states):
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
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    n_discrete_states : int
        number of discrete states (M)

    Returns
    -------
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased free energies
    """
    conf_energies = _np.zeros(shape=(n_discrete_states,), dtype=_np.float64)
    _get_conf_energies(
        <double*> _np.PyArray_DATA(log_therm_state_counts),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(bias_energy_sequence),
        <int*> _np.PyArray_DATA(conf_state_sequence),
        bias_energy_sequence.shape[0],
        n_discrete_states,
        bias_energy_sequence.shape[1],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(scratch_T),
        <double*> _np.PyArray_DATA(conf_energies))
    return conf_energies

def estimate(therm_state_counts, bias_energy_sequence, maxiter=1000, maxerr=1.0E-8, therm_energies=None):
    r"""
    Estimate the unbiased reduced free energies and thermodynamic free energies
        
    Parameters
    ----------
    therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.intc)
        numbers of samples in the T thermodynamic states
    bias_energy_sequence : numpy.ndarray(shape=(T, X), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic states for all X samples
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced free energies of the T thermodynamic states

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    """
    T = therm_state_counts.shape[0]
    log_therm_state_counts = _np.log(therm_state_counts)
    if therm_energies is None:
        therm_energies = _np.zeros(shape=(T,), dtype=_np.float64)
    old_therm_energies = therm_energies.copy()
    scratch = _np.zeros(shape=(T,), dtype=_np.float64)
    stop = False
    for _m in range(maxiter):
        update_therm_energies(log_therm_state_counts, old_therm_energies, bias_energy_sequence, scratch, therm_energies)
        if _np.max(_np.abs((therm_energies - old_therm_energies))) < maxerr:
            stop = True
        else:
            old_therm_energies[:] = therm_energies[:]
        if stop:
            break
    normalize(log_therm_state_counts, bias_energy_sequence, scratch, therm_energies)
    return therm_energies
