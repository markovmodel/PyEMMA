# This file is part of PyEMMA.
#
# Copyright (c) 2016, 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import numpy as _np
from pyemma.util import types

__all__ = [
    'get_averaged_bias_matrix',
    'get_umbrella_sampling_data',
    'get_multi_temperature_data',
    'assign_unbiased_state_label']

# ==================================================================================================
# helpers for discrete estimations
# ==================================================================================================

def get_averaged_bias_matrix(bias_sequences, dtrajs, nstates=None):
    r"""
    Computes a bias matrix via an exponential average of the observed frame wise bias energies.

    Parameters
    ----------
    bias_sequences : list of numpy.ndarray(T_i, num_therm_states)
        A single reduced bias energy trajectory or a list of reduced bias energy trajectories.
        For every simulation frame seen in trajectory i and time step t, btrajs[i][t, k] is the
        reduced bias energy of that frame evaluated in the k'th thermodynamic state (i.e. at
        the k'th Umbrella/Hamiltonian/temperature)
    dtrajs : list of numpy.ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes
        in 0,...,num_conf_states-1 enumerating the num_conf_states Markov states or the bins the
        trajectory is in at any time.
    nstates : int, optional, default=None
        Number of configuration states.

    Returns
    -------
    bias_matrix : numpy.ndarray(shape=(num_therm_states, num_conf_states)) object
        bias_energies_full[j, i] is the bias energy in units of kT for each discrete state i
        at thermodynamic state j.
    """
    from pyemma.thermo.extensions.util import (logsumexp as _logsumexp, logsumexp_pair as _logsumexp_pair)

    nmax = int(_np.max([dtraj.max() for dtraj in dtrajs]))
    if nstates is None:
        nstates = nmax + 1
    elif nstates < nmax + 1:
        raise ValueError("nstates is smaller than the number of observed microstates")
    nthermo = bias_sequences[0].shape[1]
    bias_matrix = -_np.ones(shape=(nthermo, nstates), dtype=_np.float64) * _np.inf
    counts = _np.zeros(shape=(nstates,), dtype=_np.intc)
    for s in range(len(bias_sequences)):
        for i in range(nstates):
            idx = (dtrajs[s] == i)
            nidx = idx.sum()
            if nidx == 0:
                continue
            counts[i] += nidx
            selected_bias_sequence = bias_sequences[s][idx, :]
            for k in range(nthermo):
                bias_matrix[k, i] = _logsumexp_pair(
                    bias_matrix[k, i],
                    _logsumexp(
                        _np.ascontiguousarray(-selected_bias_sequence[:, k]),
                        inplace=False))
    idx = counts.nonzero()
    log_counts = _np.log(counts[idx])
    bias_matrix *= -1.0
    bias_matrix[:, idx] += log_counts[_np.newaxis, :]
    return bias_matrix

# ==================================================================================================
# helpers for umbrella sampling simulations
# ==================================================================================================

def _ensure_umbrella_center(candidate, dimension):
    try:
        candidate = _np.asarray(candidate).astype(_np.float64).reshape((-1,))
    except ValueError:
        raise ValueError("Umbrella center " + str(candidate) + " cannot be cast as numpy.ndarray")
    if candidate.shape[0] == 1 and dimension > 1:
        return candidate[0] * _np.ones(shape=(dimension,), dtype=_np.float64)
    elif candidate.shape[0] != dimension:
        raise ValueError("Unmatching dimensions: umbrella center " + str(candidate) + \
            " is not compatible with dimension %d" % dimension)
    return candidate

def _ensure_force_constant(candidate, dimension):
    try:
        candidate = _np.asarray(candidate).astype(_np.float64)
    except ValueError:
        raise ValueError("Force constant " + str(candidate) + " cannot be cast as numpy.ndarray")
    if candidate.ndim == 0:
        candidate = candidate * _np.ones(shape=(dimension,), dtype=_np.float64)
    if candidate.shape[0] != dimension:
        raise ValueError("Force constant " + str(candidate) + \
            " has shape[0]=%d instead of %d" % (candidate.shape[0], dimension))
    if candidate.ndim == 1:
        _candidate = _np.zeros(shape=(dimension, dimension), dtype=_np.float64)
        for i, x in enumerate(candidate):
            _candidate[i, i] = x
        candidate = _candidate
    elif candidate.ndim == 2:
        if candidate.shape[0] != dimension:
            raise ValueError("Force constant " + str(candidate) + \
                " has shape[1]=%d instead of %d" % (candidate.shape[1], dimension))
    else:
        raise ValueError("Force constant " + str(candidate) + " must be a 2d numpy.ndarray")
    return candidate

def _get_umbrella_sampling_parameters(
    us_trajs, us_centers, us_force_constants, md_trajs=None, kT=None):
    umbrella_centers = []
    force_constants = []
    ttrajs = []
    nthermo = 0
    unbiased_state = None
    dimension = None
    for i, traj in enumerate(us_trajs):
        state = None
        try:
            _dimension = traj.shape[1]
        except IndexError:
            _dimension = 1
        if dimension is None:
            dimension = _dimension
        else:
            if dimension != _dimension:
                raise ValueError(
                    "Trajectory %i has unmatching dimension %d!=%d" % (i, _dimension, dimension))
        this_center = _ensure_umbrella_center(
            us_centers[i], dimension)
        this_force_constant = _ensure_force_constant(
            us_force_constants[i], dimension)
        if _np.all(this_force_constant == 0.0):
            this_center *= 0.0
        for j in range(nthermo):
            if _np.all(umbrella_centers[j] == this_center) and \
                _np.all(force_constants[j] == this_force_constant):
                state = j
                break
        if state is None:
            umbrella_centers.append(this_center.copy())
            force_constants.append(this_force_constant.copy())
            ttrajs.append(nthermo * _np.ones(shape=(traj.shape[0],), dtype=_np.intc))
            nthermo += 1
        else:
            ttrajs.append(state * _np.ones(shape=(traj.shape[0],), dtype=_np.intc))
    if md_trajs is not None:
        if not isinstance(md_trajs, (list, tuple)):
            md_trajs = [md_trajs]
        this_center = umbrella_centers[-1] * 0.0
        this_force_constant = force_constants[-1] * 0.0
        for j in range(nthermo):
            if _np.all(force_constants[j] == this_force_constant):
                unbiased_state = j
                break
        if unbiased_state is None:
            umbrella_centers.append(this_center.copy())
            force_constants.append(this_force_constant.copy())
            unbiased_state = nthermo
            nthermo += 1
        for traj in md_trajs:
            ttrajs.append(unbiased_state * _np.ones(shape=(traj.shape[0],), dtype=_np.intc))
    umbrella_centers = _np.array(umbrella_centers, dtype=_np.float64)
    force_constants = _np.array(force_constants, dtype=_np.float64)
    if kT is not None:
        if not isinstance(kT, (int, float)):
            raise ValueError("kT has wrong type:" + str(type(kT)))
        if kT <= 0.0:
            raise ValueError("non-positive kT: %f" % kT)
        force_constants /= kT
    return ttrajs, umbrella_centers, force_constants, unbiased_state

def _get_umbrella_bias_sequences(trajs, umbrella_centers, force_constants, width):
    from pyemma.thermo.extensions.util import get_umbrella_bias as _get_umbrella_bias
    bias_sequences = []
    if not isinstance(umbrella_centers, _np.ndarray):
        raise TypeError("umbrella_centers is not a numpy.ndarray: " + str(type(umbrella_centers)))
    if not isinstance(force_constants, _np.ndarray):
        raise TypeError("force_constants is not a numpy.ndarray: " + str(type(force_constants)))
    if umbrella_centers.ndim != 2:
        raise ValueError("umbrella_centers is not a 2d numpy.ndarray: " + \
            str(umbrella_centers.shape))
    if force_constants.ndim != 3:
        raise ValueError("force_constants is not a 3d numpy.ndarray: " + \
            str(force_constants.shape))
    if umbrella_centers.shape[0] != force_constants.shape[0]:
        raise ValueError("Unmatching number of umbrella centers and force constants: %d != %d" % (
            umbrella_centers.shape[0], force_constants.shape[0]))
    dimension = umbrella_centers.shape[1]
    if force_constants.shape[1] != dimension or force_constants.shape[2] != dimension:
        raise ValueError("Dimension of force_constants does not match dimension of " + \
            "umbrella_centers: %d != %d,%d" % (dimension,
                force_constants.shape[1], force_constants.shape[2]))
    for i, traj in enumerate(trajs):
        if not isinstance(traj, _np.ndarray):
            raise TypeError("Trajectory %d is not a numpy.ndarray: " % i + str(type(traj)))
        if traj.ndim == 1:
            traj = traj.reshape((-1, 1))
        if traj.shape[1] != dimension:
            raise ValueError("Trajectory %d has unmatching dimension: %d!=%d" % (
                i, traj.shape[1], dimension))
        bias_sequences.append(_get_umbrella_bias(
                _np.require(traj, requirements='C'),
                _np.require(umbrella_centers, requirements='C'),
                _np.require(force_constants, requirements='C'),
                _np.require(width, requirements='C')
        ))
    return bias_sequences

def get_umbrella_sampling_data(
    us_trajs, us_centers, us_force_constants, md_trajs=None, kT=None, width=None):
    r"""
    Wraps umbrella sampling data or a mix of umbrella sampling and and direct molecular dynamics.

    Parameters
    ----------
    us_trajs : list of N arrays, each of shape (T_i, d)
        List of arrays, each having T_i rows, one for each time step, and d columns where d is the
        dimension in which umbrella sampling was applied. Often d=1, and thus us_trajs will
        be a list of 1d-arrays.
    us_centers : array-like of size N
        List or array of N center positions. Each position must be a d-dimensional vector. For 1d
        umbrella sampling, one can simply pass a list of centers, e.g. [-5.0, -4.0, -3.0, ... ].
    us_force_constants : float or array-like of float
        The force constants used in the umbrellas, unit-less (e.g. kT per length unit). If different
        force constants were used for different umbrellas, a list or array of N force constants
        can be given. For multidimensional umbrella sampling, the force matrix must be used.
    md_trajs : list of M arrays, each of shape (T_i, d), optional, default=None
        Unbiased molecular dynamics simulations. Format like umbrella_trajs.
    kT : float (optinal)
        Use this attribute if the supplied force constants are NOT unit-less.
    width : array-like of float, optional, default=None
        Specify periodicity for individual us_traj dimensions. Each positive entry will make the
        corresponding feature periodic and use the given value as width. None/zero values will be
        treated as non-periodic.

    Returns
    -------
    ttrajs : list of N+M int arrays, each of shape (T_i,)
        The integers are indexes in 0,...,K-1 enumerating the thermodynamic states the trajectories
        are in at any time.
    btrajs : list of N+M float arrays, each of shape (T_i, K)
        The floats are the reduced bias energies for each thermodynamic state and configuration.
    umbrella_centers : float array of shape (K, d)
        The individual umbrella centers labelled accordingly to ttrajs.
    force_constants : float array of shape (K, d, d)
        The individual force matrices labelled accordingly to ttrajs.
    unbiased_state : int or None
        Index of the unbiased thermodynamic state (if present).
    """
    ttrajs, umbrella_centers, force_constants, unbiased_state = _get_umbrella_sampling_parameters(
        us_trajs, us_centers, us_force_constants, md_trajs=md_trajs, kT=kT)
    if md_trajs is None:
        md_trajs = []
    if width is None:
        width = _np.zeros(shape=(umbrella_centers.shape[1],), dtype=_np.float64)
    else:
        width = _np.fromiter(
            map(lambda w: w if w is not None and w > 0.0 else 0.0, width),
            dtype=_np.float64)
    if width.shape[0] != umbrella_centers.shape[1]:
        raise ValueError('Unmatching number of width components.')
    btrajs = _get_umbrella_bias_sequences(
        us_trajs + md_trajs, umbrella_centers, force_constants, width)
    return ttrajs, btrajs, umbrella_centers, force_constants, unbiased_state

# ==================================================================================================
# helpers for multi-temperature simulations
# ==================================================================================================

def _get_multi_temperature_parameters(temptrajs):
    temperatures = []
    for temptraj in temptrajs:
        temperatures += _np.unique(temptraj).tolist()
    temperatures = _np.array(_np.unique(temperatures), dtype=_np.float64)
    nthermo = temperatures.shape[0]
    ttrajs = []
    for temptraj in temptrajs:
        ttraj = _np.zeros(shape=temptraj.shape, dtype=_np.intc)
        for k in range(nthermo):
            ttraj[(temptraj == temperatures[k])] = k
        ttrajs.append(ttraj.copy())
    return ttrajs, temperatures

boltzmann_constant_in_kcal_per_mol = 0.0019872041
conversion_factor_J_per_cal = 4.184
conversion_shift_Celsius_to_Kelvin = 273.15

def _get_multi_temperature_bias_sequences(
    energy_trajs, temp_trajs, temperatures, reference_temperature,
    energy_unit, temp_unit):
    assert isinstance(energy_unit, str), 'energy_unit must be type str'
    assert isinstance(temp_unit, str), 'temp_unit must be type str'
    assert energy_unit.lower() in ('kcal/mol', 'kj/mol', 'kt'), \
        'energy_unit must be \'kcal/mol\', \'kJ/mol\' or \'kT\''
    assert temp_unit.lower() in ('kt', 'k', 'c'), \
        'temp_unit must be \'K\', \'C\' or \'kT\''
    btrajs = []
    if energy_unit.lower() == 'kt':
        # reduced case: energy_trajs in kT, temp_trajs unit does not matter as it cancels
        for energy_traj, temp_traj in zip(energy_trajs, temp_trajs):
            btrajs.append(
                (1.0 / temperatures[_np.newaxis, :] - 1.0 / reference_temperature) * \
                (temp_traj * energy_traj)[:, _np.newaxis])
    elif temp_unit.lower() == 'kt':
        # non-reduced case with kT values instead of temperatures
        # this implicitly assumes the users' unit of k_B equals unit of energy_trajs
        for energy_traj, temp_traj in zip(energy_trajs, temp_trajs):
            btrajs.append(
                (1.0 / temperatures[_np.newaxis, :] - 1.0 / reference_temperature) \
                    * energy_traj[:, _np.newaxis])
    else:
        # non-reduced case and temperatures given
        kT = temperatures.copy()
        rT = reference_temperature
        if temp_unit.lower() == 'c':
            kT += conversion_shift_Celsius_to_Kelvin
            rT += conversion_shift_Celsius_to_Kelvin
        kT *= boltzmann_constant_in_kcal_per_mol
        rT *= boltzmann_constant_in_kcal_per_mol
        if energy_unit.lower() == 'kj/mol':
            kT *= conversion_factor_J_per_cal
            rT *= conversion_factor_J_per_cal
        for energy_traj, temp_traj in zip(energy_trajs, temp_trajs):
            btrajs.append((1.0 / kT[_np.newaxis, :] - 1.0 / rT) * energy_traj[:, _np.newaxis])
    return btrajs

def get_multi_temperature_data(
    energy_trajs, temp_trajs, energy_unit, temp_unit, reference_temperature=None):
    r"""
    Wraps data from multi-temperature molecular dynamics.

    Parameters
    ----------
    energy_trajs : list of N arrays, each of shape (T_i,)
        List of arrays, each having T_i rows, one for each time step, containing the potential
        energies time series in units of kT, kcal/mol or kJ/mol.
    temp_trajs : list of N int arrays, each of shape (T_i,)
        List of arrays, each having T_i rows, one for each time step, containing the heat bath
        temperature time series (at which temperatures the frames were created) in units of K or C.
        Alternatively, these trajectories may contain kT values instead of temperatures.
    energy_unit: str, optional, default='kcal/mol'
        The physical unit used for energies. Current options: kcal/mol, kJ/mol, kT.
    temp_unit : str, optional, default='K'
        The physical unit used for the temperature. Current options: K, C, kT
    reference_temperature : float or None, optional, default=None
        Reference temperature against which the bias energies are computed. If not given, the lowest
        temperature or kT value is used. If given, this parameter must have the same unit as the
        temp_trajs.

    Returns
    -------
    ttrajs : list of N+M int arrays, each of shape (T_i,)
        The integers are indexes in 0,...,K-1 enumerating the thermodynamic states the trajectories
        are in at any time.
    btrajs : list of N+M float arrays, each of shape (T_i, K)
        The floats are the reduced bias energies for each thermodynamic state and configuration.
    temperatures : float array of length K
        The individual temperatures labelled accordingly to ttrajs.
    unbiased_state : int or None
        Index of the unbiased thermodynamic state (if present).
    """
    ttrajs, temperatures = _get_multi_temperature_parameters(temp_trajs)
    if reference_temperature is None:
        reference_temperature = temperatures.min()
    else:
        assert isinstance(reference_temperature, (int, float)), \
            'reference_temperature must be numeric'
        assert reference_temperature > 0.0, 'reference_temperature must be positive'
    btrajs = _get_multi_temperature_bias_sequences(
        energy_trajs, temp_trajs, temperatures, reference_temperature, energy_unit, temp_unit)
    if reference_temperature in temperatures:
        unbiased_state = _np.where(temperatures == reference_temperature)[0]
        try:
            unbiased_state = unbiased_state[0]
        except IndexError:
            unbiased_state = None
    else:
        unbiased_state = None
    return ttrajs, btrajs, temperatures, unbiased_state

# ==================================================================================================
# helpers for marking the unbiased state
# ==================================================================================================

def assign_unbiased_state_label(memm_list, unbiased_state):
    r"""
    Sets the msm label for the given list of estimated MEMM objects.

    Parameters
    ----------
    memm_list : list of estimated MEMM objects
        The MEMM objects which shall have the msm label set.
    unbiased_state : int or None
        Index of the unbiased thermodynamic state (if present).
    """
    if unbiased_state is None:
        return
    for memm in memm_list:
        assert 0 <= unbiased_state < len(memm.models), "invalid state: " + str(unbiased_state)
        memm._unbiased_state = unbiased_state
