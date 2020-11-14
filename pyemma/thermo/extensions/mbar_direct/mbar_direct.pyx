# This file is part of thermotools.
#
# Copyright 2015, 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from warnings import warn as _warn
from msmtools.util.exceptions import NotConvergedWarning as _NotConvergedWarning

from . import mbar as _mbar
from .callback import CallbackInterrupt

__all__ = [
    'update_therm_weights',
    'estimate_therm_energies',
    'estimate']

cdef extern from "_mbar_direct.h":
    void _mbar_direct_update_therm_weights(
        int *therm_state_counts, double *therm_weights, double *bias_weight_sequences,
        int n_therm_states, int seq_length, double *new_therm_weights)

def update_therm_weights(
    _np.ndarray[int, ndim=1, mode="c"] therm_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_weights not None,
    bias_weight_sequences, # _np.ndarray[double, ndim=2, mode="c"]
    _np.ndarray[double, ndim=1, mode="c"] new_therm_weights not None):
    r"""
    Calculate the reduced thermodynamic free energies therm_energies
        
    Parameters
    ----------
    therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.intc)
        state counts in each of the T thermodynamic states
    therm_weights : numpy.ndarray(shape=(T), dtype=numpy.float64)
        probabilities of the T thermodynamic states
    bias_weight_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
        bias weights in the T thermodynamic states for all X samples
    new_therm_weights : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the probabilities of the T thermodynamic states
    """
    new_therm_weights[:] = 0.0
    for i in range(len(bias_weight_sequences)):
        _mbar_direct_update_therm_weights(
            <int*> _np.PyArray_DATA(therm_state_counts),
            <double*> _np.PyArray_DATA(therm_weights),
            <double*> _np.PyArray_DATA(bias_weight_sequences[i]),
            therm_state_counts.shape[0],
            bias_weight_sequences[i].shape[0],
            <double*> _np.PyArray_DATA(new_therm_weights))
    new_therm_weights /= new_therm_weights[0]

def estimate_therm_energies(
    therm_state_counts, bias_energy_sequences,
    maxiter=1000, maxerr=1.0E-8, therm_energies=None,
    save_convergence_info=0, callback=None):
    r"""
    Estimate the thermodynamic free energies
        
    Parameters
    ----------
    therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.intc)
        numbers of samples in the T thermodynamic states
    bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic states for all X samples
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced free energies of the T thermodynamic states
    save_convergence_info : int, optional
        every save_convergence_info iteration steps, store the actual increment

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    increments : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of increments
    """
    T = therm_state_counts.shape[0]
    therm_state_counts = therm_state_counts.astype(_np.intc)
    log_therm_state_counts = _np.log(therm_state_counts)
    shift = _np.min([_np.min(b, axis=0) for b in bias_energy_sequences], axis=0)
    if therm_energies is None:
        therm_energies = _np.zeros(shape=(T,), dtype=_np.float64)
        therm_weights = _np.ones(shape=(T,), dtype=_np.float64)
    else:
        therm_weights = _np.exp(shift - therm_energies)
    bias_weight_sequences = [_np.exp(shift - b) for b in bias_energy_sequences]
    old_therm_energies = therm_energies.copy()
    old_therm_weights = therm_weights.copy()
    increments = []
    sci_count = 0
    for m in range(maxiter):
        sci_count += 1
        update_therm_weights(
            therm_state_counts, old_therm_weights, bias_weight_sequences, therm_weights)
        therm_energies = -_np.log(therm_weights)
        delta_therm_energies = _np.abs(therm_energies - old_therm_energies)
        err = _np.max(delta_therm_energies)
        if sci_count == save_convergence_info:
            sci_count = 0
            increments.append(err)
        if callback is not None:
            try:
                callback(iteration_step = m,
                         therm_weights = therm_weights,
                         old_therm_weights = old_therm_weights,
                         err=err,
                         maxerr=maxerr,
                         maxiter=maxiter)
            except CallbackInterrupt:
                break
        if err < maxerr:
            break
        else:
            old_therm_weights[:] = therm_weights[:]
            old_therm_energies[:] = therm_energies[:]
    therm_energies = shift - _np.log(therm_weights)
    if err >= maxerr:
        _warn("MBAR did not converge: last increment = %.5e" % err, _NotConvergedWarning)
    if save_convergence_info == 0:
        increments = None
    else:
        increments = _np.array(increments, dtype=_np.float64)
    return therm_energies, increments

def estimate(
    therm_state_counts, bias_energy_sequences, conf_state_sequences,
    maxiter=1000, maxerr=1.0E-8, therm_energies=None,
    n_conf_states=None, save_convergence_info=0, callback=None):
    r"""
    Estimate the (un)biased reduced free energies and thermodynamic free energies
        
    Parameters
    ----------
    therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.intc)
        numbers of samples in the T thermodynamic states
    bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic states for all X samples
    conf_state_sequences : list of numpy.ndarray(shape=(X_i), dtype=numpy.intc)
        discrete state indices (cluster indices) for all X samples
    maxiter : int
        maximum number of iterations
    maxerr : float
        convergence criterion based on absolute change in free energies
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64), OPTIONAL
        initial guess for the reduced free energies of the T thermodynamic states
    n_conf_states : int, optional, default=None
        the number of configurational states in `conf_state_sequence`.
        If None, this is set to max(conf_state_sequence)+1.
    save_convergence_info : int, optional
        every save_convergence_info iteration steps, store the actual increment

    Returns
    -------
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased discrete state (cluster) free energies
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced discrete state free energies for all combinations of
        T thermodynamic states and M discrete states
    increments : numpy.ndarray(dtype=numpy.float64, ndim=1)
        stored sequence of increments
    """
    T = therm_state_counts.shape[0]
    therm_state_counts = therm_state_counts.astype(_np.intc)
    if n_conf_states is None:
        M = 1 + max([_np.max(c) for c in conf_state_sequences])
    else:
        M = n_conf_states
    assert len(conf_state_sequences) == len(bias_energy_sequences)
    for s, b in zip(conf_state_sequences, bias_energy_sequences):
        assert s.ndim == 1
        assert s.dtype == _np.intc
        assert b.ndim == 2
        assert b.dtype == _np.float64
        assert s.shape[0] == b.shape[0]
        assert b.shape[1] == T
        assert s.flags.c_contiguous
        assert b.flags.c_contiguous
    log_therm_state_counts = _np.log(therm_state_counts)
    shift = _np.min([_np.min(b, axis=0) for b in bias_energy_sequences], axis=0)
    if therm_energies is None:
        therm_energies = _np.zeros(shape=(T,), dtype=_np.float64)
        therm_weights = _np.ones(shape=(T,), dtype=_np.float64)
    else:
        therm_weights = _np.exp(shift - therm_energies)
    bias_weight_sequences = [_np.exp(shift - b) for b in bias_energy_sequences]
    scratch_M = _np.zeros(shape=(M,), dtype=_np.float64)
    scratch_T = _np.zeros(shape=(T,), dtype=_np.float64)
    therm_energies, increments = estimate_therm_energies(
        therm_state_counts, bias_energy_sequences,
        maxiter=maxiter, maxerr=maxerr, therm_energies=therm_energies,
        save_convergence_info=save_convergence_info, callback=callback)
    conf_energies, biased_conf_energies = _mbar.get_conf_energies(
        log_therm_state_counts, therm_energies,
        bias_energy_sequences, conf_state_sequences, scratch_T, M)
    _mbar.normalize(
        scratch_M, therm_energies, conf_energies, biased_conf_energies)
    return therm_energies, conf_energies, biased_conf_energies, increments
