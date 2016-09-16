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
from .util import assign_unbiased_state_label as _assign_unbiased_state_label

__docformat__ = "restructuredtext en"
__author__ = "Frank Noe, Christoph Wehmeyer"
__copyright__ = "Copyright 2015, 2016, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Frank Noe", "Christoph Wehmeyer"]
__maintainer__ = "Christoph Wehmeyer"
__email__ = "christoph.wehmeyer@fu-berlin.de"

__all__ = [
    'estimate_umbrella_sampling',
    'estimate_multi_temperature',
    'tram',
    'dtram',
    'wham',
    'mbar']

# ==================================================================================================
# wrappers for specific simulation types
# ==================================================================================================

def estimate_umbrella_sampling(
    us_trajs, us_dtrajs, us_centers, us_force_constants, md_trajs=None, md_dtrajs=None, kT=None,
    maxiter=10000, maxerr=1.0E-15, save_convergence_info=0,
    estimator='wham', lag=1, dt_traj='1 step', init=None, init_maxiter=10000, init_maxerr=1.0E-8,
    **kwargs):
    r"""
    This function acts as a wrapper for ``tram()``, ``dtram()``, and ``wham()`` and handles the
    calculation of bias energies (``bias``) and thermodynamic state trajectories (``ttrajs``)
    when the data comes from umbrella sampling and (optional) unbiased simulations.

    Parameters
    ----------
    us_trajs : list of N arrays, each of shape (T_i, d)
        List of arrays, each having T_i rows, one for each time step, and d columns where d is the
        dimensionality of the subspace in which umbrella sampling was applied. Often d=1, and thus
        us_trajs will be a list of 1d-arrays.
    us_dtrajs : list of N int arrays, each of shape (T_i,)
        The integers are indexes in 0,...,n-1 enumerating the n discrete states or the bins the
        umbrella sampling trajectory is in at any time.
    us_centers : list of N floats or d-dimensional arrays of floats
        List or array of N center positions. Each position must be a d-dimensional vector. For 1d
        umbrella sampling, one can simply pass a list of centers, e.g. [-5.0, -4.0, -3.0, ... ].
    us_force_constants : list of N floats or d- or dxd-dimensional arrays of floats
        The force constants used in the umbrellas, unit-less (e.g. kT per squared length unit). For
        multidimensional umbrella sampling, the force matrix must be used.
    md_trajs : list of M arrays, each of shape (T_i, d), optional, default=None
        Unbiased molecular dynamics simulations; format like us_trajs.
    md_dtrajs : list of M int arrays, each of shape (T_i,)
        The integers are indexes in 0,...,n-1 enumerating the n discrete states or the bins the
        unbiased trajectory is in at any time.
    kT : float or None, optional, default=None
        Use this attribute if the supplied force constants are NOT unit-less; kT must have the same
        energy unit as the force constants.
    maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1.0E-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.
    save_convergence_info : int, optional, default=0
        Every save_convergence_info iteration steps, store the actual increment
        and the actual loglikelihood; 0 means no storage.
    estimator : str, optional, default='wham'
        Specify one of the available estimators

        | 'wham':   use WHAM
        | 'mbar':   use MBAR
        | 'dtram':  use the discrete version of TRAM
        | 'tram':  use TRAM
    lag : int or list of int, optional, default=1
        Integer lag time at which transitions are counted. Providing a list of lag times will
        trigger one estimation per lag time.
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
        Use a specific initialization for the self-consistent iteration:

        | None:    use a hard-coded guess for free energies and Lagrangian multipliers
        | 'wham':  perform a short WHAM estimate to initialize the free energies (only with dtram)
        | 'mbar':  perform a short MBAR estimate to initialize the free energies (only with tram)
    init_maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations during the initialization.
    init_maxerr : float, optional, default=1.0E-8
        Convergence criterion for the initialization.
    **kwargs : dict, optional
        You can use this to pass estimator-specific named parameters to the chosen estimator, which
        are not already coverd by ``estimate_umbrella_sampling()``.

    Returns
    -------
    estimator_obj : MEMM or MultiThermModel or list thereof
        The requested estimator/model object, i.e., WHAM, MBAR, DTRAM or TRAM. If multiple lag times
        are given, a list of objects is returned (one MEMM per lag time).

    Example
    -------
    We look at a 1D umbrella sampling simulation with two umbrellas at 1.1 and 1.3 on the reaction
    coordinate with spring constant of 1.0; additionally, we have two unbiased simulations.

    We start with a joint clustering and use TRAM for the estimation:

    >>> from pyemma.coordinates import cluster_regspace as regspace
    >>> from pyemma.thermo import estimate_umbrella_sampling as estimate_us
    >>> import numpy as np
    >>> us_centers = [1.1, 1.3]
    >>> us_force_constants = [1.0, 1.0]
    >>> us_trajs = [np.array([1.0, 1.1, 1.2, 1.1, 1.0, 1.1]).reshape((-1, 1)), np.array([1.3, 1.2, 1.3, 1.4, 1.4, 1.3]).reshape((-1, 1))]
    >>> md_trajs = [np.array([0.9, 1.0, 1.1, 1.2, 1.3, 1.4]).reshape((-1, 1)), np.array([1.5, 1.4, 1.3, 1.4, 1.4, 1.5]).reshape((-1, 1))]
    >>> cluster = regspace(data=us_trajs+md_trajs, max_centers=10, dmin=0.15)
    >>> us_dtrajs = cluster.dtrajs[:2]
    >>> md_dtrajs = cluster.dtrajs[2:]
    >>> centers = cluster.clustercenters
    >>> tram = estimate_us(us_trajs, us_dtrajs, us_centers, us_force_constants, md_trajs=md_trajs, md_dtrajs=md_dtrajs, estimator='tram', lag=1)
    >>> tram.f # doctest: +ELLIPSIS
    array([ 0.63...,  1.60...,  1.31...])

    """
    from .util import get_umbrella_sampling_data as _get_umbrella_sampling_data
    # sanity checks
    if estimator not in ['wham', 'mbar', 'dtram', 'tram']:
        raise ValueError("unsupported estimator: %s" % estimator)
    if not isinstance(us_trajs, (list, tuple)):
        raise ValueError("The parameter us_trajs must be a list of numpy.ndarray objects")
    if not isinstance(us_centers, (list, tuple)):
        raise ValueError(
            "The parameter us_centers must be a list of floats or numpy.ndarray objects")
    if not isinstance(us_force_constants, (list, tuple)):
        raise ValueError(
            "The parameter us_force_constants must be a list of floats or numpy.ndarray objects")
    if len(us_trajs) != len(us_centers):
        raise ValueError("Unmatching number of umbrella sampling trajectories and centers: %d!=%d" \
            % (len(us_trajs), len(us_centers)))
    if len(us_trajs) != len(us_force_constants):
        raise ValueError(
            "Unmatching number of umbrella sampling trajectories and force constants: %d!=%d" \
                % (len(us_trajs), len(us_force_constants)))
    if len(us_trajs) != len(us_dtrajs):
            raise ValueError(
                "Number of continuous and discrete umbrella sampling trajectories does not " + \
                "match: %d!=%d" % (len(us_trajs), len(us_dtrajs)))
    i = 0
    for traj, dtraj in zip(us_trajs, us_dtrajs):
        if traj.shape[0] != dtraj.shape[0]:
            raise ValueError(
                "Lengths of continuous and discrete umbrella sampling trajectories with " + \
                "index %d does not match: %d!=%d" % (i, len(us_trajs), len(us_dtrajs)))
        i += 1
    if md_trajs is not None:
        if not isinstance(md_trajs, (list, tuple)):
            raise ValueError("The parameter md_trajs must be a list of numpy.ndarray objects")
        if md_dtrajs is None:
            raise ValueError("You have provided md_trajs, but md_dtrajs is None")
    if md_dtrajs is None:
        md_dtrajs = []
    else:
        if md_trajs is None:
            raise ValueError("You have provided md_dtrajs, but md_trajs is None")
        if len(md_trajs) != len(md_dtrajs):
            raise ValueError(
                "Number of continuous and discrete unbiased trajectories does not " + \
                "match: %d!=%d" % (len(md_trajs), len(md_dtrajs)))
        i = 0
        for traj, dtraj in zip(md_trajs, md_dtrajs):
            if traj.shape[0] != dtraj.shape[0]:
                raise ValueError(
                    "Lengths of continuous and discrete unbiased trajectories with " + \
                    "index %d does not match: %d!=%d" % (i, len(md_trajs), len(md_dtrajs)))
            i += 1
    # data preparation
    ttrajs, btrajs, umbrella_centers, force_constants, unbiased_state = _get_umbrella_sampling_data(
        us_trajs, us_centers, us_force_constants, md_trajs=md_trajs, kT=kT)
    estimator_obj = None
    # estimation
    if estimator == 'wham':
        estimator_obj = wham(
            ttrajs, us_dtrajs + md_dtrajs,
            _get_averaged_bias_matrix(btrajs, us_dtrajs + md_dtrajs),
            maxiter=maxiter, maxerr=maxerr,
            save_convergence_info=save_convergence_info, dt_traj=dt_traj)
    elif estimator == 'mbar':
        allowed_keys = ['direct_space']
        parsed_kwargs = dict([(i, kwargs[i]) for i in allowed_keys if i in kwargs])
        estimator_obj = mbar(
            ttrajs, us_dtrajs + md_dtrajs, btrajs,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, **parsed_kwargs)
    elif estimator == 'dtram':
        allowed_keys = ['count_mode', 'connectivity']
        parsed_kwargs = dict([(i, kwargs[i]) for i in allowed_keys if i in kwargs])
        estimator_obj = dtram(
            ttrajs, us_dtrajs + md_dtrajs,
            _get_averaged_bias_matrix(btrajs, us_dtrajs + md_dtrajs),
            lag, unbiased_state=unbiased_state,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, init=init, init_maxiter=init_maxiter, init_maxerr=init_maxerr,
            **parsed_kwargs)
    elif estimator == 'tram':
        allowed_keys = [
            'count_mode', 'connectivity', 'connectivity_factor','nn',
            'direct_space', 'N_dtram_accelerations', 'equilibrium']
        parsed_kwargs = dict([(i, kwargs[i]) for i in allowed_keys if i in kwargs])
        estimator_obj = tram(
            ttrajs, us_dtrajs + md_dtrajs, btrajs, lag, unbiased_state=unbiased_state,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, init=init, init_maxiter=init_maxiter, init_maxerr=init_maxerr,
            **parsed_kwargs)
    # adding thermodynamic state information and return results
    try:
        estimator_obj.umbrella_centers = umbrella_centers
        estimator_obj.force_constants = force_constants
    except AttributeError:
        for obj in estimator_obj:
            obj.umbrella_centers = umbrella_centers
            obj.force_constants = force_constants
    return estimator_obj


def estimate_multi_temperature(
    energy_trajs, temp_trajs, dtrajs,
    energy_unit='kcal/mol', temp_unit='K', reference_temperature=None,
    maxiter=10000, maxerr=1.0E-15, save_convergence_info=0,
    estimator='wham', lag=1, dt_traj='1 step', init=None, init_maxiter=10000, init_maxerr=1e-8,
    **kwargs):
    r"""
    This function acts as a wrapper for ``tram()``, ``dtram()``, and ``wham()`` and handles the
    calculation of bias energies (``bias``) and thermodynamic state trajectories (``ttrajs``)
    when the data comes from multi-temperature simulations.

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
    reference_temperature : float or None, optional, default=None
        Reference temperature against which the bias energies are computed. If not given, the lowest
        temperature or kT value is used. If given, this parameter must have the same unit as the
        temp_trajs.
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
        | 'mbar':   use MBAR
        | 'dtram':  use the discrete version of TRAM
        | 'tram':  use TRAM
    lag : int or list of int, optional, default=1
        Integer lag time at which transitions are counted. Providing a list of lag times will
        trigger one estimation per lag time.
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
        Use a specific initialization for the self-consistent iteration:

        | None:    use a hard-coded guess for free energies and Lagrangian multipliers
        | 'wham':  perform a short WHAM estimate to initialize the free energies (only with dtram)
        | 'mbar':  perform a short MBAR estimate to initialize the free energies (only with tram)
    init_maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations during the initialization.
    init_maxerr : float, optional, default=1.0E-8
        Convergence criterion for the initialization.
    **kwargs : dict, optional
        You can use this to pass estimator-specific named parameters to the chosen estimator, which
        are not already coverd by ``estimate_multi_temperature()``.

    Returns
    -------
    estimator_obj : MEMM or MultiThermModel or list thereof
        The requested estimator/model object, i.e., WHAM, MBAR, DTRAM or TRAM. If multiple lag times
        are given, a list of objects is returned (one MEMM per lag time).

    Example
    -------
    We look at 1D simulations at two different kT values 1.0 and 2.0, already clustered data, and
    we use TRAM for the estimation:

    >>> from pyemma.thermo import estimate_multi_temperature as estimate_mt
    >>> import numpy as np
    >>> energy_trajs = [np.array([1.6, 1.4, 1.0, 1.0, 1.2, 1.0, 1.0]), np.array([0.8, 0.7, 0.5, 0.6, 0.7, 0.8, 0.7])]
    >>> temp_trajs = [np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])]
    >>> dtrajs = [np.array([0, 1, 2, 2, 2, 2, 2]), np.array([0, 1, 2, 2, 1, 0, 1])]
    >>> tram = estimate_mt(energy_trajs, temp_trajs, dtrajs, energy_unit='kT', temp_unit='kT', estimator='tram', lag=1)
    >>> tram.f # doctest: +ELLIPSIS
    array([ 2.90...,  1.72...,  0.26...])

    Note that alhough we only used one temperature per trajectory, ``estimate_multi_temperature()``
    can handle temperature changes as well.

    """
    if estimator not in ['wham', 'mbar', 'dtram', 'tram']:
        ValueError("unsupported estimator: %s" % estimator)
    from .util import get_multi_temperature_data as _get_multi_temperature_data
    ttrajs, btrajs, temperatures, unbiased_state = _get_multi_temperature_data(
        energy_trajs, temp_trajs, energy_unit, temp_unit,
        reference_temperature=reference_temperature)
    estimator_obj = None
    if estimator == 'wham':
        estimator_obj = wham(
            ttrajs, dtrajs,
            _get_averaged_bias_matrix(btrajs, dtrajs),
            maxiter=maxiter, maxerr=maxerr,
            save_convergence_info=save_convergence_info, dt_traj=dt_traj)
    elif estimator == 'mbar':
        allowed_keys = ['direct_space']
        parsed_kwargs = dict([(i, kwargs[i]) for i in allowed_keys if i in kwargs])
        estimator_obj = mbar(
            ttrajs, dtrajs, btrajs,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, **parsed_kwargs)
    elif estimator == 'dtram':
        allowed_keys = ['count_mode', 'connectivity']
        parsed_kwargs = dict([(i, kwargs[i]) for i in allowed_keys if i in kwargs])
        estimator_obj = dtram(
            ttrajs, dtrajs,
            _get_averaged_bias_matrix(btrajs, dtrajs),
            lag, unbiased_state=unbiased_state,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, init=init, init_maxiter=init_maxiter, init_maxerr=init_maxerr,
            **parsed_kwargs)
    elif estimator == 'tram':
        allowed_keys = [
            'count_mode', 'connectivity', 'connectivity_factor','nn',
            'direct_space', 'N_dtram_accelerations', 'equilibrium']
        parsed_kwargs = dict([(i, kwargs[i]) for i in allowed_keys if i in kwargs])
        estimator_obj = tram(
            ttrajs, dtrajs, btrajs, lag, unbiased_state=unbiased_state,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, init=init, init_maxiter=init_maxiter, init_maxerr=init_maxerr,
            **parsed_kwargs)
    try:
        estimator_obj.temperatures = temperatures
    except AttributeError:
        for obj in estimator_obj:
            obj.temperatures = temperatures
    return estimator_obj

# ==================================================================================================
# wrappers for the estimators
# ==================================================================================================

def tram(
    ttrajs, dtrajs, bias, lag, unbiased_state=None,
    count_mode='sliding', connectivity='summed_count_matrix',
    maxiter=10000, maxerr=1.0E-15, save_convergence_info=0, dt_traj='1 step',
    connectivity_factor=1.0, nn=None, direct_space=False, N_dtram_accelerations=0, callback=None,
    init='mbar', init_maxiter=10000, init_maxerr=1e-8, equilibrium=None):
    r"""
    Transition-based reweighting analysis method

    Parameters
    ----------
    ttrajs : numpy.ndarray(T), or list of numpy.ndarray(T_i)
        A single discrete trajectory or a list of discrete trajectories. The integers are
        indexes in 0,...,num_therm_states-1 enumerating the thermodynamic states the trajectory is
        in at any time.
    dtrajs : numpy.ndarray(T) of int, or list of numpy.ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes
        in 0,...,num_conf_states-1 enumerating the num_conf_states Markov states or the bins the
        trajectory is in at any time.
    bias : numpy.ndarray(T, num_therm_states), or list of numpy.ndarray(T_i, num_therm_states)
        A single reduced bias energy trajectory or a list of reduced bias energy trajectories.
        For every simulation frame seen in trajectory i and time step t, btrajs[i][t, k] is the
        reduced bias energy of that frame evaluated in the k'th thermodynamic state (i.e. at
        the k'th umbrella/Hamiltonian/temperature)
    lag : int or list of int, optional, default=1
        Integer lag time at which transitions are counted. Providing a list of lag times will
        trigger one estimation per lag time.
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

    connectivity : str, optional, default='summed_count_matrix'
        One of 'summed_count_matrix', 'strong_in_every_ensemble',
        'neighbors', 'post_hoc_RE' or 'BAR_variance'.
        Defines what should be considered a connected set in the joint space
        of conformations and thermodynamic ensembles.
        For details see thermotools.cset.compute_csets_TRAM.
    nn : int, optional, default=None
        Only needed if connectivity='neighbors'
        See thermotools.cset.compute_csets_TRAM.
    connectivity_factor : float, optional, default=1.0
        Only needed if connectivity='post_hoc_RE' or 'BAR_variance'. Weakens the connectivity
        requirement, see thermotools.cset.compute_csets_TRAM.
    direct_space : bool, optional, default=False
        Whether to perform the self-consitent iteration with Boltzmann factors
        (direct space) or free energies (log-space). When analyzing data from
        multi-temperature simulations, direct-space is not recommended.
    N_dtram_accelerations : int, optional, default=0
        Convergence of TRAM can be speeded up by interleaving the updates
        in the self-consitent iteration with a dTRAM-like update step.
        N_dtram_accelerations says how many times the dTRAM-like update
        step should be applied in every iteration of the TRAM equations.
        Currently this is only effective if direct_space=True.
    init : str, optional, default=None
        Use a specific initialization for self-consistent iteration:

        | None:    use a hard-coded guess for free energies and Lagrangian multipliers
        | 'wham':  perform a short WHAM estimate to initialize the free energies

    init_maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations during the initialization.
    init_maxerr : float, optional, default=1.0E-8
        Convergence criterion for the initialization.

    Returns
    -------
    tram_estimators : MEMM or list of MEMMs
        A multi-ensemble Markov state model (for each given lag time) which consists of stationary
        and kinetic quantities at all temperatures/thermodynamic states.

    Example
    -------
    **Umbrella sampling**: Suppose we simulate in K umbrellas, centered at
    positions :math:`y_0,...,y_{K-1}` with bias energies

    .. math::
        b_k(x) = \frac{c_k}{2 \textrm{kT}} \cdot (x - y_k)^2

    Suppose we have one simulation of length T in each umbrella, and they are ordered from 0 to K-1.
    We have discretized the x-coordinate into 100 bins.
    Then dtrajs and ttrajs should each be a list of :math:`K` arrays.
    dtrajs would look for example like this::

    [ (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, ...),  (0, 1, 0, 1, 0, 1, 1, 0, 0, 1, ...), ... ]

    where each array has length T, and is the sequence of bins (in the range 0 to 99) visited along
    the trajectory. ttrajs would look like this::

    [ (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...),  (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...), ... ]

    Because trajectory 1 stays in umbrella 1 (index 0), trajectory 2 stays in umbrella 2 (index 1),
    and so forth.

    The bias would be a list of :math:`T \times K` arrays which specify each frame's bias energy in
    all thermodynamic states:

    [ ((0, 1.7, 2.3, 6.1, ...), ...), ((0, 2.4, 3.1, 9,5, ...), ...), ... ]

    Let us try the above example:

    >>> from pyemma.thermo import tram
    >>> import numpy as np
    >>> ttrajs = [np.array([0,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1])]
    >>> dtrajs = [np.array([0,0,0,0,1,1,1]), np.array([0,1,0,1,0,1,1])]
    >>> bias = [np.array([[1,0],[1,0],[0,0],[0,0],[0,0],[0,0],[0,0]],dtype=np.float64), np.array([[1,0],[0,0],[0,0],[1,0],[0,0],[1,0],[1,0]],dtype=np.float64)]
    >>> tram_obj = tram(ttrajs, dtrajs, bias, 1)
    >>> tram_obj.log_likelihood() # doctest: +ELLIPSIS
    -29.111...
    >>> tram_obj.count_matrices # doctest: +SKIP
    array([[[1 1]
            [0 4]]
           [[0 3]
            [2 1]]], dtype=int32)
    >>> tram_obj.stationary_distribution # doctest: +ELLIPSIS
    array([ 0.38...  0.61...])

    References
    ----------

    .. [1] Wu, H. et al 2016
        Multiensemble Markov models of molecular thermodynamics and kinetics
        Proc. Natl. Acad. Sci. USA 113 E3221--E3230

    """
    # prepare trajectories
    ttrajs = _types.ensure_dtraj_list(ttrajs)
    dtrajs = _types.ensure_dtraj_list(dtrajs)
    if len(ttrajs) != len(dtrajs):
        raise ValueError("Unmatching number of dtraj/ttraj elements: %d!=%d" % (
            len(dtrajs), len(ttrajs)))
    if len(ttrajs) != len(bias):
        raise ValueError("Unmatching number of ttraj/bias elements: %d!=%d" % (
            len(ttrajs), len(bias)))
    for ttraj, dtraj, btraj in zip(ttrajs, dtrajs, bias):
        if len(ttraj) != len(dtraj):
            raise ValueError("Unmatching number of data points in ttraj/dtraj: %d!=%d" % (
                len(ttraj), len(dtraj)))
        if len(ttraj) != btraj.shape[0]:
            raise ValueError("Unmatching number of data points in ttraj/bias trajectory: %d!=%d" % (
                len(ttraj), len(btraj)))
    # check lag time(s)
    lags = _np.asarray(lag, dtype=_np.intc).reshape((-1,)).tolist()
    # build TRAM and run estimation
    from pyemma.thermo import TRAM as _TRAM
    tram_estimators = [
        _TRAM(
            _lag, count_mode=count_mode, connectivity=connectivity,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, connectivity_factor=connectivity_factor, nn=nn,
            direct_space=direct_space, N_dtram_accelerations=N_dtram_accelerations,
            callback=callback, init='mbar', init_maxiter=init_maxiter, init_maxerr=init_maxerr,
            equilibrium=equilibrium).estimate((ttrajs, dtrajs, bias)) for _lag in lags]
    _assign_unbiased_state_label(tram_estimators, unbiased_state)
    # return
    if len(tram_estimators) == 1:
        return tram_estimators[0]
    return tram_estimators

def dtram(
    ttrajs, dtrajs, bias, lag, unbiased_state=None,
    count_mode='sliding', connectivity='largest',
    maxiter=10000, maxerr=1.0E-15, save_convergence_info=0, dt_traj='1 step',
    init=None, init_maxiter=10000, init_maxerr=1.0E-8):
    r"""
    Discrete transition-based reweighting analysis method

    Parameters
    ----------
    ttrajs : numpy.ndarray(T) of int, or list of numpy.ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are
        indexes in 0,...,num_therm_states-1 enumerating the thermodynamic states the trajectory is
        in at any time.
    dtrajs : numpy.ndarray(T) of int, or list of numpy.ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes
        in 0,...,num_conf_states-1 enumerating the num_conf_states Markov states or the bins the
        trajectory is in at any time.
    bias : numpy.ndarray(shape=(num_therm_states, num_conf_states)) object
        bias_energies_full[j, i] is the bias energy in units of kT for each discrete state i
        at thermodynamic state j.
    lag : int or list of int, optional, default=1
        Integer lag time at which transitions are counted. Providing a list of lag times will
        trigger one estimation per lag time.
    count_mode : str, optional, default='sliding'
        Mode to obtain count matrices from discrete trajectories. Should be one of:

        * 'sliding' : a trajectory of length T will have :math:`T-\tau` counts at time indexes
            .. math::
                 (0 \rightarrow \tau), (1 \rightarrow \tau+1), ..., (T-\tau-1 \rightarrow T-1)
        * 'sample' : a trajectory of length T will have :math:`T/\tau` counts at time indexes
            .. math::
                    (0 \rightarrow \tau), (\tau \rightarrow 2 \tau), ..., ((T/\tau-1) \tau \rightarrow T)

        Currently only 'sliding' is supported.
    connectivity : str, optional, default='largest'
        Defines what should be considered a connected set in the joint space of conformations and
        thermodynamic ensembles. Currently only 'largest' is supported.
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

    init_maxiter : int, optional, default=10000
        The maximum number of self-consistent iterations during the initialization.
    init_maxerr : float, optional, default=1.0E-8
        Convergence criterion for the initialization.

    Returns
    -------
    dtram_estimators : MEMM or list of MEMMs
        A multi-ensemble Markov state model (for each given lag time) which consists of stationary
        and kinetic quantities at all temperatures/thermodynamic states.

    Example
    -------
    **Umbrella sampling**: Suppose we simulate in K umbrellas, centered at
    positions :math:`y_0,...,y_{K-1}` with bias energies

    .. math::
        b_k(x) = \frac{c_k}{2 \textrm{kT}} \cdot (x - y_k)^2

    Suppose we have one simulation of length T in each umbrella, and they are ordered from 0 to K-1.
    We have discretized the x-coordinate into 100 bins.
    Then dtrajs and ttrajs should each be a list of :math:`K` arrays.
    dtrajs would look for example like this::

    [ (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, ...),  (0, 1, 0, 1, 0, 1, 1, 0, 0, 1, ...), ... ]

    where each array has length T, and is the sequence of bins (in the range 0 to 99) visited along
    the trajectory. ttrajs would look like this::

    [ (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...),  (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...), ... ]

    Because trajectory 1 stays in umbrella 1 (index 0), trajectory 2 stays in umbrella 2 (index 1),
    and so forth. bias is a :math:`K \times n` matrix with all reduced bias energies evaluated at
    all centers:

    .. math::
        \left(\begin{array}{cccc}
            b_0(y_0) &  b_0(y_1) &  ... &  b_0(y_{n-1}) \\
            b_1(y_0) &  b_1(y_1) &  ... &  b_1(y_{n-1}) \\
            ... \\
            b_{K-1}(y_0) &  b_{K-1}(y_1) &  ... &  b_{K-1}(y_{n-1})
        \end{array}\right)

    Let us try the above example:

    >>> from pyemma.thermo import dtram
    >>> import numpy as np
    >>> ttrajs = [np.array([0,0,0,0,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1,1,1,1])]
    >>> dtrajs = [np.array([0,0,0,0,1,1,1,0,0,0]), np.array([0,1,0,1,0,1,1,0,0,1])]
    >>> bias = np.array([[0.0, 0.0], [0.5, 1.0]])
    >>> dtram_obj = dtram(ttrajs, dtrajs, bias, 1)
    >>> dtram_obj.log_likelihood() # doctest: +ELLIPSIS
    -9.805...
    >>> dtram_obj.count_matrices # doctest: +SKIP
    array([[[5, 1],
            [1, 2]],
           [[1, 4],
            [3, 1]]], dtype=int32)
    >>> dtram_obj.stationary_distribution # doctest: +ELLIPSIS
    array([ 0.38...,  0.61...])

    References
    ----------

    .. [1] Wu, H. et al 2014
        Statistically optimal analysis of state-discretized trajectory data from multiple thermodynamic states
        J. Chem. Phys. 141, 214106

    """
    # prepare trajectories
    ttrajs = _types.ensure_dtraj_list(ttrajs)
    dtrajs = _types.ensure_dtraj_list(dtrajs)
    if len(ttrajs) != len(dtrajs):
        raise ValueError("Unmatching number of dtraj/ttraj elements: %d!=%d" % (
            len(dtrajs), len(ttrajs)) )
    for ttraj, dtraj in zip(ttrajs, dtrajs):
        if len(ttraj) != len(dtraj):
            raise ValueError("Unmatching number of data points in ttraj/dtraj: %d!=%d" % (
                len(ttraj), len(dtraj)))
    # check lag time(s)
    lags = _np.asarray(lag, dtype=_np.intc).reshape((-1,)).tolist()
    # build DTRAM and run estimation
    from pyemma.thermo import DTRAM
    dtram_estimators = [
        DTRAM(
            bias, _lag,
            count_mode=count_mode, connectivity=connectivity,
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            dt_traj=dt_traj, init=init, init_maxiter=init_maxiter,
            init_maxerr=init_maxerr).estimate((ttrajs, dtrajs)) for _lag in lags]
    _assign_unbiased_state_label(dtram_estimators, unbiased_state)
    # return
    if len(dtram_estimators) == 1:
        return dtram_estimators[0]
    return dtram_estimators

def wham(
    ttrajs, dtrajs, bias,
    maxiter=100000, maxerr=1.0E-15, save_convergence_info=0, dt_traj='1 step'):
    r"""
    Weighted histogram analysis method

    Parameters
    ----------
    ttrajs : numpy.ndarray(T) of int, or list of numpy.ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are
        indexes in 0,...,num_therm_states-1 enumerating the thermodynamic states the trajectory is
        in at any time.
    dtrajs : numpy.ndarray(T) of int, or list of numpy.ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes
        in 0,...,num_conf_states-1 enumerating the num_conf_states Markov states or the bins the
        trajectory is in at any time.
    bias : numpy.ndarray(shape=(num_therm_states, num_conf_states)) object
        bias_energies_full[j, i] is the bias energy in units of kT for each discrete state i
        at thermodynamic state j.
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
    wham_estimator : MultiThermModel
        A stationary model which consists of thermodynamic quantities at all
        temperatures/thermodynamic states.

    Example
    -------
    **Umbrella sampling**: Suppose we simulate in K umbrellas, centered at
    positions :math:`y_0,...,y_{K-1}` with bias energies

    .. math::
        b_k(x) = \frac{c_k}{2 \textrm{kT}} \cdot (x - y_k)^2

    Suppose we have one simulation of length T in each umbrella, and they are ordered from 0 to K-1.
    We have discretized the x-coordinate into 100 bins.
    Then dtrajs and ttrajs should each be a list of :math:`K` arrays.
    dtrajs would look for example like this::
    
    [ (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, ...),  (0, 1, 0, 1, 0, 1, 1, 0, 0, 1, ...), ... ]
    
    where each array has length T, and is the sequence of bins (in the range 0 to 99) visited along
    the trajectory. ttrajs would look like this::

    [ (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...),  (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...), ... ]
    
    Because trajectory 1 stays in umbrella 1 (index 0), trajectory 2 stays in umbrella 2 (index 1),
    and so forth. bias is a :math:`K \times n` matrix with all reduced bias energies evaluated at
    all centers:

    .. math::
        \left(\begin{array}{cccc}
            b_0(y_0) &  b_0(y_1) &  ... &  b_0(y_{n-1}) \\
            b_1(y_0) &  b_1(y_1) &  ... &  b_1(y_{n-1}) \\
            ... \\
            b_{K-1}(y_0) &  b_{K-1}(y_1) &  ... &  b_{K-1}(y_{n-1})
        \end{array}\right)

    Let us try the above example:

    >>> from pyemma.thermo import wham
    >>> import numpy as np
    >>> ttrajs = [np.array([0,0,0,0,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1,1,1,1])]
    >>> dtrajs = [np.array([0,0,0,0,1,1,1,0,0,0]), np.array([0,1,0,1,0,1,1,0,0,1])]
    >>> bias = np.array([[0.0, 0.0], [0.5, 1.0]])
    >>> wham_obj = wham(ttrajs, dtrajs, bias)
    >>> wham_obj.log_likelihood() # doctest: +ELLIPSIS
    -6.6...
    >>> wham_obj.state_counts # doctest: +SKIP
    array([[7, 3],
           [5, 5]])
    >>> wham_obj.stationary_distribution # doctest: +ELLIPSIS +REPORT_NDIFF
    array([ 0.5...,  0.4...])

    References
    ----------
    
    .. [1] Ferrenberg, A.M. and Swensen, R.H. 1988.
        New Monte Carlo Technique for Studying Phase Transitions.
        Phys. Rev. Lett. 23, 2635--2638

    .. [2] Kumar, S. et al 1992.
        The Weighted Histogram Analysis Method for Free-Energy Calculations on Biomolecules. I. The Method.
        J. Comp. Chem. 13, 1011--1021

    """
    # check trajectories
    ttrajs = _types.ensure_dtraj_list(ttrajs)
    dtrajs = _types.ensure_dtraj_list(dtrajs)
    if len(ttrajs) != len(dtrajs):
        raise ValueError("Unmatching number of dtraj/ttraj elements: %d!=%d" % (
            len(dtrajs), len(ttrajs)) )
    for ttraj, dtraj in zip(ttrajs, dtrajs):
        if len(ttraj) != len(dtraj):
            raise ValueError("Unmatching number of data points in ttraj/dtraj: %d!=%d" % (
                len(ttraj), len(dtraj)))
    # build WHAM
    from pyemma.thermo import WHAM
    wham_estimator = WHAM(
        bias,
        maxiter=maxiter, maxerr=maxerr,
        save_convergence_info=save_convergence_info, dt_traj=dt_traj)
    # run estimation
    return wham_estimator.estimate((ttrajs, dtrajs))

def mbar(
    ttrajs, dtrajs, bias,
    maxiter=100000, maxerr=1.0E-15, save_convergence_info=0,
    dt_traj='1 step', direct_space=False):
    r"""
    Multi-state Bennet acceptance ratio

    Parameters
    ----------
    ttrajs : numpy.ndarray(T) of int, or list of numpy.ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are
        indexes in 0,...,num_therm_states-1 enumerating the thermodynamic states the trajectory is
        in at any time.
    dtrajs : numpy.ndarray(T) of int, or list of numpy.ndarray(T_i) of int
        A single discrete trajectory or a list of discrete trajectories. The integers are indexes
        in 0,...,num_conf_states-1 enumerating the num_conf_states Markov states or the bins the
        trajectory is in at any time.
    bias : numpy.ndarray(T, num_therm_states), or list of numpy.ndarray(T_i, num_therm_states)
        A single reduced bias energy trajectory or a list of reduced bias energy trajectories.
        For every simulation frame seen in trajectory i and time step t, btrajs[i][t, k] is the
        reduced bias energy of that frame evaluated in the k'th thermodynamic state (i.e. at
        the k'th umbrella/Hamiltonian/temperature)
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

    direct_space : bool, optional, default=False
        Whether to perform the self-consitent iteration with Boltzmann factors
        (direct space) or free energies (log-space). When analyzing data from
        multi-temperature simulations, direct-space is not recommended.

    Returns
    -------
    mbar_estimator : MultiThermModel
        A stationary model which consists of thermodynamic quantities at all
        temperatures/thermodynamic states.

    Example
    -------
    **Umbrella sampling**: Suppose we simulate in K umbrellas, centered at
    positions :math:`y_0,...,y_{K-1}` with bias energies

    .. math::
        b_k(x) = \frac{c_k}{2 \textrm{kT}} \cdot (x - y_k)^2

    Suppose we have one simulation of length T in each umbrella, and they are ordered from 0 to K-1.
    We have discretized the x-coordinate into 100 bins.
    Then dtrajs and ttrajs should each be a list of :math:`K` arrays.
    dtrajs would look for example like this::

    [ (0, 0, 0, 0, 1, 1, 1, 0, 0, 0, ...),  (0, 1, 0, 1, 0, 1, 1, 0, 0, 1, ...), ... ]

    where each array has length T, and is the sequence of bins (in the range 0 to 99) visited along
    the trajectory. ttrajs would look like this::

    [ (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ...),  (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...), ... ]

    Because trajectory 1 stays in umbrella 1 (index 0), trajectory 2 stays in umbrella 2 (index 1),
    and so forth.

    The bias would be a list of :math:`T \times K` arrays which specify each frame's bias energy in
    all thermodynamic states:

    [ ((0, 1.7, 2.3, 6.1, ...), ...), ((0, 2.4, 3.1, 9,5, ...), ...), ... ]

    Let us try the above example:

    >>> from pyemma.thermo import mbar
    >>> import numpy as np
    >>> ttrajs = [np.array([0,0,0,0,0,0,0]), np.array([1,1,1,1,1,1,1])]
    >>> dtrajs = [np.array([0,0,0,0,1,1,1]), np.array([0,1,0,1,0,1,1])]
    >>> bias = [np.array([[1,0],[1,0],[0,0],[0,0],[0,0],[0,0],[0,0]],dtype=np.float64), np.array([[1,0],[0,0],[0,0],[1,0],[0,0],[1,0],[1,0]],dtype=np.float64)]
    >>> mbar_obj = mbar(ttrajs, dtrajs, bias, maxiter=1000000, maxerr=1.0E-14)
    >>> mbar_obj.stationary_distribution # doctest: +ELLIPSIS
    array([ 0.5...  0.5...])

    References
    ----------
    
    .. [1] Shirts, M.R. and Chodera, J.D. 2008
        Statistically optimal analysis of samples from multiple equilibrium states
        J. Chem. Phys. 129, 124105

    """
    # check trajectories
    ttrajs = _types.ensure_dtraj_list(ttrajs)
    dtrajs = _types.ensure_dtraj_list(dtrajs)
    if len(ttrajs) != len(dtrajs):
        raise ValueError("Unmatching number of dtraj/ttraj elements: %d!=%d" % (
            len(dtrajs), len(ttrajs)))
    if len(ttrajs) != len(bias):
        raise ValueError("Unmatching number of ttraj/bias elements: %d!=%d" % (
            len(ttrajs), len(bias)))
    for ttraj, dtraj, btraj in zip(ttrajs, dtrajs, bias):
        if len(ttraj) != len(dtraj):
            raise ValueError("Unmatching number of data points in ttraj/dtraj: %d!=%d" % (
                len(ttraj), len(dtraj)))
        if len(ttraj) != btraj.shape[0]:
            raise ValueError("Unmatching number of data points in ttraj/bias trajectory: %d!=%d" % (
                len(ttraj), len(btraj)))
    # build MBAR
    from pyemma.thermo import MBAR
    mbar_estimator = MBAR(
        maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
        dt_traj=dt_traj, direct_space=direct_space)
    # run estimation
    return mbar_estimator.estimate((ttrajs, dtrajs, bias))
