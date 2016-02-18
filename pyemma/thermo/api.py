# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
from pyemma.util import types as _types
from .util import get_averaged_bias_matrix as _get_averaged_bias_matrix

__docformat__ = "restructuredtext en"
__author__ = "Frank Noe, Christoph Wehmeyer"
__copyright__ = "Copyright 2015, 2016, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Frank Noe", "Christoph Wehmeyer"]
__maintainer__ = "Christoph Wehmeyer"
__email__ = "christoph.wehmeyer@fu-berlin.de"

__all__ = [
    'estimate_umbrella_sampling',
    'estimate_multi_temperature',
    'dtram',
    'wham']

# ==================================================================================================
# wrappers for specific simulation types
# ==================================================================================================

def estimate_umbrella_sampling(
    us_trajs, us_dtrajs, us_centers, us_force_constants, md_trajs=None, md_dtrajs=None, kT=None,
    maxiter=10000, maxerr=1.0E-15, save_convergence_info=0,
    estimator='wham', lag=1, dt_traj='1 step', init=None):
    r"""
    Wraps umbrella sampling data or a mix of umbrella sampling and and direct molecular dynamics.

    Parameters
    ----------
    us_trajs : list of N arrays, each of shape (T_i, d)
        List of arrays, each having T_i rows, one for each time step, and d columns where d is the
        dimension in which umbrella sampling was applied. Often d=1, and thus us_trajs will
        be a list of 1d-arrays.
    us_dtrajs : list of N int arrays, each of shape (T_i,)
        The integers are indexes in 0,...,n-1 enumerating the n discrete states or the bins the
        trajectory is in at any time.
    us_centers : array-like of size N
        List or array of N center positions. Each position must be a d-dimensional vector. For 1d
        umbrella sampling, one can simply pass a list of centers, e.g. [-5.0, -4.0, -3.0, ... ].
    us_force_constants : float or array-like of float
        The force constants used in the umbrellas, unit-less (e.g. kT per length unit). If different
        force constants were used for different umbrellas, a list or array of N force constants
        can be given. For multidimensional umbrella sampling, the force matrix must be used.
    md_trajs : list of M arrays, each of shape (T_i, d), optional, default=None
        Unbiased molecular dynamics simulations. Format like umbrella_trajs.
    md_dtrajs : list of M int arrays, each of shape (T_i,)
        The integers are indexes in 0,...,n-1 enumerating the n discrete states or the bins the
        trajectory is in at any time.
    kT : float (optinal)
        Use this attribute if the supplied force constants are NOT unit-less.
    maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1E-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    save_convergence_info : int, optional, default=0
        Every save_convergence_info iteration steps, store the actual increment
        and the actual loglikelihood; 0 means no storage.
    estimator : str, optional, default='wham'
        Specify one of the available estimators

        | 'wham':   use WHAM
        | 'dtram':  use the discrete version of TRAM
    lag : int, optional, default=1
        Integer lag time at which transitions are counted.
    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis
        algorithms such as plotting tools to pretty-print the axes. By default '1 step', i.e.
        there is no physical time unit.  Specify by a number, whitespace and unit. Permitted
        units are (* is an arbitrary string):

        |  'fs',   'femtosecond*'
        |  'ps',   'picosecond*'
        |  'ns',   'nanosecond*'
        |  'us',   'microsecond*'
        |  'ms',   'millisecond*'
        |  's',    'second*'
    init : str, optional, default=None
        Use a specific initialization for self-consistent iteration:

        | None:    use a hard-coded guess for free energies and Lagrangian multipliers
        | 'wham':  perform a short WHAM estimate to initialize the free energies

    Returns
    -------
    _estimator : MEMM or StationaryModel
        The requested estimator/model object, i.e., WHAM or DTRAM.
    """
    assert estimator in ['wham', 'dtram'], "unsupported estimator: %s" % estimator
    from .util import get_umbrella_sampling_data as _get_umbrella_sampling_data
    ttrajs, btrajs, umbrella_centers, force_constants = _get_umbrella_sampling_data(
        us_trajs, us_centers, us_force_constants, md_trajs=md_trajs, kT=kT)
    if md_dtrajs is None:
        md_dtrajs = []
    _estimator = None
    if estimator == 'wham':
        _estimator = wham(
            ttrajs, us_dtrajs + md_dtrajs,
            _get_averaged_bias_matrix(btrajs, us_dtrajs + md_dtrajs),
            maxiter=maxiter, maxerr=maxerr,
            save_convergence_info=save_convergence_info, dt_traj=dt_traj)
    elif estimator == 'dtram':
        _estimator = dtram(
            ttrajs, us_dtrajs + md_dtrajs,
            _get_averaged_bias_matrix(btrajs, us_dtrajs + md_dtrajs),
            lag,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, init=init)
    _estimator.umbrella_centers = umbrella_centers
    _estimator.force_constants = force_constants
    return _estimator


def estimate_multi_temperature(
    energy_trajs, temp_trajs, dtrajs, reference_temperature=None,
    energy_unit='kcal/mol', temp_unit='K',
    maxiter=10000, maxerr=1.0E-15, save_convergence_info=0,
    estimator='wham', lag=1, dt_traj='1 step', init=None):
    r"""
    Wraps multi-temperature data.

    Parameters
    ----------
    energy_trajs : list of N arrays, each of shape (T_i,)
        List of arrays, each having T_i rows, one for each time step, containing the potential
        energies time series in units of kT, kcal/mol or kJ/mol.
    temp_trajs : list of N int arrays, each of shape (T_i,)
        List of arrays, each having T_i rows, one for each time step, containing the heat bath
        temperature time series (at which temperatures the frames were created) in units of K or C.
        Alternatively, these trajectories may contain kT values instead of temperatures.
    dtrajs : list of N int arrays, each of shape (T_i,)
        The integers are indexes in 0,...,n-1 enumerating the n discrete states or the bins the
        trajectory is in at any time.
    energy_unit: str, optional, default='kcal/mol'
        The physical unit used for energies. Current options: kcal/mol, kJ/mol, kT.
    temp_unit : str, optional, default='K'
        The physical unit used for the temperature. Current options: K, C, kT
    maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1E-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    save_convergence_info : int, optional, default=0
        Every save_convergence_info iteration steps, store the actual increment
        and the actual loglikelihood; 0 means no storage.
    estimator : str, optional, default='wham'
        Specify one of the available estimators

        | 'wham':   use WHAM
        | 'dtram':  use the discrete version of TRAM
    lag : int, optional, default=1
        Integer lag time at which transitions are counted.
    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis
        algorithms such as plotting tools to pretty-print the axes. By default '1 step', i.e.
        there is no physical time unit.  Specify by a number, whitespace and unit. Permitted
        units are (* is an arbitrary string):

        |  'fs',   'femtosecond*'
        |  'ps',   'picosecond*'
        |  'ns',   'nanosecond*'
        |  'us',   'microsecond*'
        |  'ms',   'millisecond*'
        |  's',    'second*'
    init : str, optional, default=None
        Use a specific initialization for self-consistent iteration:

        | None:    use a hard-coded guess for free energies and Lagrangian multipliers
        | 'wham':  perform a short WHAM estimate to initialize the free energies

    Returns
    -------
    _estimator : MEMM or StationaryModel
        The requested estimator/model object, i.e., WHAM or DTRAM.
    """
    assert estimator in ['wham', 'dtram'], "unsupported estimator: %s" % estimator
    from .util import get_multi_temperature_data as _get_multi_temperature_data
    ttrajs, btrajs, temperatures = _get_multi_temperature_data(
        energy_trajs, temp_trajs, energy_unit, temp_unit,
        reference_temperature=reference_temperature)
    _estimator = None
    if estimator == 'wham':
        _estimator = wham(
            ttrajs, dtrajs,
            _get_averaged_bias_matrix(btrajs, dtrajs),
            maxiter=maxiter, maxerr=maxerr,
            save_convergence_info=save_convergence_info, dt_traj=dt_traj)
    elif estimator == 'dtram':
        _estimator = dtram(
            ttrajs, dtrajs,
            _get_averaged_bias_matrix(btrajs, dtrajs),
            lag,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, init=init)
    _estimator.temperatures = temperatures
    return _estimator

# ==================================================================================================
# wrappers for the estimators
# ==================================================================================================

def dtram(
    ttrajs, dtrajs, bias, lag,
    maxiter=10000, maxerr=1.0E-15, save_convergence_info=0, dt_traj='1 step', init=None):
    r"""
    Discrete transition-based reweighting analysis method

    Parameters
    ----------
    ttrajs : ndarray(T) of int, or list of ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are
        indexes in 0,...,K-1 enumerating the thermodynamic states the trajectory is in at any time.
    dtrajs : ndarray(T) of int, or list of ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes
        in 1,...,n enumerating the n Markov states or the bins the trajectory is in at any time.
    bias : ndarray(K, n)
        bias[j,i] is the bias energy for each discrete state i at thermodynamic state j.
    lag : int
        Integer lag time at which transitions are counted.
    maxiter : int, optional, default=10000
        The maximum number of dTRAM iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1e-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    save_convergence_info : int, optional, default=0
        Every save_convergence_info iteration steps, store the actual increment
        and the actual loglikelihood; 0 means no storage.
    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis
        algorithms such as plotting tools to pretty-print the axes. By default '1 step', i.e.
        there is no physical time unit.  Specify by a number, whitespace and unit. Permitted
        units are (* is an arbitrary string):

        |  'fs',   'femtosecond*'
        |  'ps',   'picosecond*'
        |  'ns',   'nanosecond*'
        |  'us',   'microsecond*'
        |  'ms',   'millisecond*'
        |  's',    'second*'
    init : str, optional, default=None
        Use a specific initialization for self-consistent iteration:

        | None:    use a hard-coded guess for free energies and Lagrangian multipliers
        | 'wham':  perform a short WHAM estimate to initialize the free energies

    Returns
    -------
    memm : MEMM
        A multi-thermodynamic Markov state model which consists of stationary and kinetic
        quantities at all temperatures/thermodynamic states.

    Example
    -------
    **Example: Umbrella sampling**. Suppose we simulate in K umbrellas, centered at
    positions :math:`y_1,...,y_K` with bias energies
    .. math::
        b_k(x) = 0.5 * c_k * (x - y_k)^2 / kT
    Suppose we have one simulation of length T in each umbrella, and they are ordered from 1 to K.
    We have discretized the x-coordinate into 100 bins.
    Then dtrajs and ttrajs should each be a list of :math:`K` arrays.
    dtrajs would look for example like this:
    [ (1, 2, 2, 3, 2, ...),  (2, 4, 5, 4, 4, ...), ... ]
    where each array has length T, and is the sequence of bins (in the range 0 to 99) visited along
    the trajectory. ttrajs would look like this:
    [ (0, 0, 0, 0, 0, ...),  (1, 1, 1, 1, 1, ...), ... ]
    Because trajectory 1 stays in umbrella 1 (index 0), trajectory 2 stays in umbrella 2 (index 1),
    and so forth. bias is a :math:`K \times n` matrix with all reduced bias energies evaluated at
    all centers:
    [[b_0(y_0), b_0(y_1), ..., b_0(y_n)],
     [b_1(y_0), b_1(y_1), ..., b_1(y_n)],
     ...
     [b_K(y_0), b_K(y_1), ..., b_K(y_n)]]
    """
    # prepare trajectories
    ttrajs = _types.ensure_dtraj_list(ttrajs)
    dtrajs = _types.ensure_dtraj_list(dtrajs)
    assert len(ttrajs) == len(dtrajs)
    X = []
    for i in range(len(ttrajs)):
        ttraj = ttrajs[i]
        dtraj = dtrajs[i]
        assert len(ttraj) == len(dtraj)
        X.append(_np.ascontiguousarray(_np.array([ttraj, dtraj]).T))
    # build DTRAM
    from pyemma.thermo import DTRAM
    dtram_estimator = DTRAM(
        bias, lag,
        count_mode='sliding', connectivity='largest',
        maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
        dt_traj=dt_traj, init=init)
    # run estimation
    return dtram_estimator.estimate(X)

def wham(
    ttrajs, dtrajs, bias,
    maxiter=100000, maxerr=1.0E-15, save_convergence_info=0, dt_traj='1 step'):
    r"""
    Weighted histogram analysis method

    Parameters
    ----------
    ttrajs : ndarray(T) of int, or list of ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are
        indexes in 1,...,K enumerating the thermodynamic states the trajectory is in at any time.
    dtrajs : ndarray(T) of int, or list of ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes
        in 1,...,n enumerating the n Markov states or the bins the trajectory is in at any time.
    bias : ndarray(K, n)
        bias[j,i] is the bias energy for each discrete state i at thermodynamic state j.
    maxiter : int, optional, default=10000
        The maximum number of dTRAM iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1e-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    save_convergence_info : int, optional, default=0
        Every save_convergence_info iteration steps, store the actual increment
        and the actual loglikelihood; 0 means no storage.
    dt_traj : str, optional, default='1 step'
        Description of the physical time corresponding to the lag. May be used by analysis
        algorithms such as plotting tools to pretty-print the axes. By default '1 step', i.e.
        there is no physical time unit.  Specify by a number, whitespace and unit. Permitted
        units are (* is an arbitrary string):

        |  'fs',   'femtosecond*'
        |  'ps',   'picosecond*'
        |  'ns',   'nanosecond*'
        |  'us',   'microsecond*'
        |  'ms',   'millisecond*'
        |  's',    'second*'

    Returns
    -------
    sm : StationaryModel
        A stationary model which consists of thermodynamic quantities at all
        temperatures/thermodynamic states.

    Example
    -------
    **Example: Umbrella sampling**. Suppose we simulate in K umbrellas, centered at
    positions :math:`y_1,...,y_K` with bias energies

    .. math::
        b_k(x) = 0.5 * c_k * (x - y_k)^2 / kT

    Suppose we have one simulation of length T in each umbrella, and they are ordered from 1 to K.
    We have discretized the x-coordinate into 100 bins.
    Then dtrajs and ttrajs should each be a list of :math:`K` arrays.
    dtrajs would look for example like this::
    
    [ (1, 2, 2, 3, 2, ...),  (2, 4, 5, 4, 4, ...), ... ]
    where each array has length T, and is the sequence of bins (in the range 0 to 99) visited along
    the trajectory. ttrajs would look like this:
    [ (0, 0, 0, 0, 0, ...),  (1, 1, 1, 1, 1, ...), ... ]
    Because trajectory 1 stays in umbrella 1 (index 0), trajectory 2 stays in umbrella 2 (index 1),
    and so forth. bias is a :math:`K \times n` matrix with all reduced bias energies evaluated at
    all centers::

    [[b_0(y_0), b_0(y_1), ..., b_0(y_n)],
     [b_1(y_0), b_1(y_1), ..., b_1(y_n)],
     ...
     [b_K(y_0), b_K(y_1), ..., b_K(y_n)]]

    """
    # prepare trajectories
    ttrajs = _types.ensure_dtraj_list(ttrajs)
    dtrajs = _types.ensure_dtraj_list(dtrajs)
    assert len(ttrajs) == len(dtrajs)
    X = []
    for i in range(len(ttrajs)):
        ttraj = ttrajs[i]
        dtraj = dtrajs[i]
        assert len(ttraj) == len(dtraj)
        X.append(_np.ascontiguousarray(_np.array([ttraj, dtraj]).T))
    # build WHAM
    from pyemma.thermo import WHAM
    wham_estimator = WHAM(
        bias,
        maxiter=maxiter, maxerr=maxerr,
        save_convergence_info=save_convergence_info, dt_traj=dt_traj)
    # run estimation
    return wham_estimator.estimate(X)
