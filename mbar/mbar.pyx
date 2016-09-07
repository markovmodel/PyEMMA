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

from .callback import CallbackInterrupt

__all__ = [
    'update_therm_energies',
    'get_conf_energies',
    'normalize',
    'get_pointwise_unbiased_free_energies',
    'estimate_therm_energies',
    'estimate']

cdef extern from "_mbar.h":
    void _mbar_update_therm_energies(
        double *log_therm_state_counts, double *therm_energies, double *bias_energy_sequence,
        int n_therm_states, int seq_length, double *scratch_T, double *new_therm_energies)
    void _mbar_get_conf_energies(
        double *log_therm_state_counts, double *therm_energies,
        double *bias_energy_sequence, int * conf_state_sequence,
        int n_therm_states, int n_conf_states, int seq_length,
        double *scratch_T, double *conf_energies, double *biased_conf_energies)
    extern void _mbar_normalize(
        int n_therm_states, int n_conf_states, double *scratch_M,
        double *therm_energies, double *conf_energies, double *biased_conf_energies)
    void _mbar_get_pointwise_unbiased_free_energies(
        int k, double *log_therm_state_counts, double *therm_energies,
        double *bias_energy_sequence,
        int n_therm_states,  int seq_length,
        double *scratch_T, double *pointwise_unbiased_free_energies)


def update_therm_energies(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    bias_energy_sequences, # _np.ndarray[double, ndim=2, mode="c"]
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    _np.ndarray[double, ndim=1, mode="c"] new_therm_energies not None):
    r"""
    Calculate the reduced thermodynamic free energies therm_energies.
        
    Parameters
    ----------
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    new_therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        target array for the reduced free energies of the T thermodynamic states
    """
    new_therm_energies[:] = _np.inf
    for i in range(len(bias_energy_sequences)):
        _mbar_update_therm_energies(
            <double*> _np.PyArray_DATA(log_therm_state_counts),
            <double*> _np.PyArray_DATA(therm_energies),
            <double*> _np.PyArray_DATA(bias_energy_sequences[i]),
            therm_energies.shape[0],
            bias_energy_sequences[i].shape[0],
            <double*> _np.PyArray_DATA(scratch_T),
            <double*> _np.PyArray_DATA(new_therm_energies))
    new_therm_energies -= new_therm_energies[0]

def get_conf_energies(
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    bias_energy_sequences, # _np.ndarray[double, ndim=2, mode="c"]
    conf_state_sequences, # _np.ndarray[int, ndim=1, mode="c"]
    _np.ndarray[double, ndim=1, mode="c"] scratch_T not None,
    n_conf_states):
    r"""
    Calculate the reduced unbiased free energies conf_energies.
        
    Parameters
    ----------
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    conf_state_sequences : list of numpy.ndarray(shape=(X_i), dtype=numpy.intc)
        discrete states indices for all X samples
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array
    n_conf_states : int
        number of discrete states (M)

    Returns
    -------
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased free energies
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic and M discrete states
    """
    conf_energies = _np.zeros(shape=(n_conf_states,), dtype=_np.float64)
    biased_conf_energies = _np.zeros(
        shape=(therm_energies.shape[0], n_conf_states), dtype=_np.float64)
    conf_energies[:] = _np.inf
    biased_conf_energies[:] = _np.inf
    for i in range(len(bias_energy_sequences)):
        _mbar_get_conf_energies(
            <double*> _np.PyArray_DATA(log_therm_state_counts),
            <double*> _np.PyArray_DATA(therm_energies),
            <double*> _np.PyArray_DATA(bias_energy_sequences[i]),
            <int*> _np.PyArray_DATA(conf_state_sequences[i]),
            therm_energies.shape[0],
            n_conf_states,
            conf_state_sequences[i].shape[0],
            <double*> _np.PyArray_DATA(scratch_T),
            <double*> _np.PyArray_DATA(conf_energies),
            <double*> _np.PyArray_DATA(biased_conf_energies))
    return conf_energies, biased_conf_energies

def normalize(
    _np.ndarray[double, ndim=1, mode="c"] scratch_M not None,
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] conf_energies not None,
    _np.ndarray[double, ndim=2, mode="c"] biased_conf_energies not None):
    r"""
    Shift the reduced thermodynamic free energies therm_energies such that the
    unbiased thermodynamic free energy is zero.
        
    Parameters
    ----------
    scratch_M : numpy.ndarray(shape=(M), dtype=numpy.float64)
        scratch array
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    conf_energies : numpy.ndarray(shape=(M), dtype=numpy.float64)
        reduced unbiased discrete state (cluster) free energies
    biased_conf_energies : numpy.ndarray(shape=(T, M), dtype=numpy.float64)
        reduced discrete state free energies for all combinations of
        T thermodynamic states and M discrete states
    """
    _mbar_normalize(
        therm_energies.shape[0],
        conf_energies.shape[0],
        <double*> _np.PyArray_DATA(scratch_M),
        <double*> _np.PyArray_DATA(therm_energies),
        <double*> _np.PyArray_DATA(conf_energies),
        <double*> _np.PyArray_DATA(biased_conf_energies))

def get_pointwise_unbiased_free_energies(
    k,
    _np.ndarray[double, ndim=1, mode="c"] log_therm_state_counts not None,
    bias_energy_sequences, # _np.ndarray[double, ndim=2, mode="c"]
    _np.ndarray[double, ndim=1, mode="c"] therm_energies not None,
    _np.ndarray[double, ndim=1, mode="c"] scratch_T,
    pointwise_unbiased_free_energies): # _np.ndarray[double, ndim=1, mode="c"]
    r'''
    Compute the pointwise free energies :math:`\mu^{k}(x)` for all x.

    Parameters
    ----------
    k : int or None
        thermodynamic state, if k is None, compute pointwise free energies
        of the unbiased ensemble.
    log_therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.float64)
        log of the state counts in each of the T thermodynamic states
    bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
        bias energies in the T thermodynamic states for all X samples
    therm_energies : numpy.ndarray(shape=(T), dtype=numpy.float64)
        reduced free energies of the T thermodynamic states
    scratch_T : numpy.ndarray(shape=(T), dtype=numpy.float64)
        scratch array for logsumexp operations
    pointwise_unbiased_free_energies : list of numpy.ndarray(shape=(X_i), dtype=numpy.float64)
        target arrays for the pointwise free energies
    '''

    if scratch_T is None:
        scratch_T = _np.zeros(log_therm_state_counts.shape[0], dtype=_np.float64)
    if k is None:
        k = -1
    assert len(bias_energy_sequences)==len(pointwise_unbiased_free_energies)
    for b, p in zip(bias_energy_sequences, pointwise_unbiased_free_energies):
        assert b.ndim == 2
        assert b.dtype == _np.float64
        assert p.ndim == 1
        assert p.dtype == _np.float64
        assert b.shape[0] == p.shape[0]
        assert b.shape[1] == log_therm_state_counts.shape[0]
        assert b.flags.c_contiguous
        assert p.flags.c_contiguous
    for i in range(len(bias_energy_sequences)):
        _mbar_get_pointwise_unbiased_free_energies(
            k,
            <double*> _np.PyArray_DATA(log_therm_state_counts),
            <double*> _np.PyArray_DATA(therm_energies),
            <double*> _np.PyArray_DATA(bias_energy_sequences[i]),
            log_therm_state_counts.shape[0],
            bias_energy_sequences[i].shape[0],
            <double*> _np.PyArray_DATA(scratch_T),
            <double*> _np.PyArray_DATA(pointwise_unbiased_free_energies[i]))

def estimate_therm_energies(
    therm_state_counts, bias_energy_sequences,
    maxiter=1000, maxerr=1.0E-8, therm_energies=None,
    n_conf_states=None, save_convergence_info=0, callback=None):
    r"""
    Estimate the thermodynamic free energies.
        
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
    n_conf_states : int, optional, default=None
        the number of configurational states in `conf_state_sequence`.
        If None, this is set to max(conf_state_sequence)+1.
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
    log_therm_state_counts = _np.log(therm_state_counts)
    if therm_energies is None:
        therm_energies = _np.zeros(shape=(T,), dtype=_np.float64)
    old_therm_energies = therm_energies.copy()
    increments = []
    sci_count = 0
    scratch = _np.zeros(shape=(T,), dtype=_np.float64)
    for m in range(maxiter):
        sci_count += 1
        update_therm_energies(
            log_therm_state_counts, old_therm_energies, bias_energy_sequences,
            scratch, therm_energies)
        delta_therm_energies = _np.abs(therm_energies - old_therm_energies)
        err = _np.max(delta_therm_energies)
        if sci_count == save_convergence_info:
            sci_count = 0
            increments.append(err)
        if callback is not None:
            try:
                callback(therm_energies=therm_energies,
                         old_therm_energies=old_therm_energies,
                         delta_therm_energies=delta_therm_energies,
                         iteration_step=m,
                         err=err,
                         maxerr=maxerr,
                         maxiter=maxiter)
            except CallbackInterrupt:
                break
        if err < maxerr:
            break
        else:
            old_therm_energies[:] = therm_energies[:]
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
    Estimate the (un)biased reduced free energies and thermodynamic free energies.
        
    Parameters
    ----------
    therm_state_counts : numpy.ndarray(shape=(T), dtype=numpy.intc)
        numbers of samples in the T thermodynamic states
    bias_energy_sequences : list of numpy.ndarray(shape=(X_i, T), dtype=numpy.float64)
        reduced bias energies in the T thermodynamic states for all X samples
    conf_state_sequences : list of numpy.ndarray(shape=(X_i), dtype=numpy.float64)
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
    if n_conf_states is None:
        M = 1 + max([_np.max(s) for s in conf_state_sequences])
    else:
        M = n_conf_states
    assert len(conf_state_sequences)==len(bias_energy_sequences)
    for s, b in zip(conf_state_sequences, bias_energy_sequences):
        assert s.ndim == 1
        assert s.dtype == _np.intc
        assert b.ndim == 2
        assert b.dtype == _np.float64
        assert s.shape[0] == b.shape[0]
        assert b.shape[1] == T
        assert s.flags.c_contiguous
        assert b.flags.c_contiguous
    increments = []
    scratch_M = _np.zeros(shape=(M,), dtype=_np.float64)
    scratch_T = _np.zeros(shape=(T,), dtype=_np.float64)
    therm_energies, increments = estimate_therm_energies(
        therm_state_counts, bias_energy_sequences,
        maxiter=maxiter, maxerr=maxerr, therm_energies=therm_energies,
        save_convergence_info=save_convergence_info, callback=callback)
    conf_energies, biased_conf_energies = get_conf_energies(
        _np.log(therm_state_counts), therm_energies, bias_energy_sequences, conf_state_sequences,
        scratch_T, M)
    normalize(scratch_M, therm_energies, conf_energies, biased_conf_energies)
    return therm_energies, conf_energies, biased_conf_energies, increments
