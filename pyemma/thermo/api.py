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
from .util import get_umbrella_sampling_data as _get_umbrella_sampling_data
from .util import get_averaged_bias_matrix as _get_averaged_bias_matrix

__docformat__ = "restructuredtext en"
__author__ = "Frank Noe, Christoph Wehmeyer"
__copyright__ = "Copyright 2015, 2016, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Frank Noe", "Christoph Wehmeyer"]
__maintainer__ = "Christoph Wehmeyer"
__email__ = "christoph.wehmeyer@fu-berlin.de"

__all__ = [
    'umbrella_sampling',
    'dtram',
    'wham']

# ===================================
# Data Loaders and Readers
# ===================================

def umbrella_sampling(
    us_trajs, us_dtrajs, us_centers, us_force_constants, md_trajs=None, md_dtrajs=None, kT=None,
    maxiter=1000, maxerr=1.0E-5, save_convergence_info=0,
    estimator='wham', lag=1, dt_traj='1 step', init=None):
    assert estimator in ['wham', 'dtram'], "unsupported estimator: %s" % estimator
    ttrajs, btrajs, umbrella_centers, force_constants = _get_umbrella_sampling_data(
        us_trajs, us_centers, us_force_constants, md_trajs=md_trajs, kT=kT)
    if md_dtrajs is None:
        md_dtrajs = []
    _estimator = None
    if estimator == 'wham':
        _estimator = wham(
            ttrajs, us_dtrajs + md_dtrajs,
            _get_averaged_bias_matrix(btrajs, us_dtrajs + md_dtrajs),
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info)
    elif estimator == 'dtram':
        _estimator = dtram(
            ttrajs, us_dtrajs + md_dtrajs,
            _get_averaged_bias_matrix(btrajs, us_dtrajs + md_dtrajs),
            maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
            lag=lag, dt_traj=dt_traj, init=init)
    _estimator.umbrella_centers = umbrella_centers
    _estimator.force_constants = force_constants
    return _estimator

# This corresponds to the source function in coordinates.api
def multitemperature_to_bias(utrajs, ttrajs, kTs):
    r""" Wraps umbrella sampling data or a mix of umbrella sampling and and direct molecular dynamics
    The probability at the thermodynamic ground state is:
    .. math:
        \pi(x) = \mathrm{e}^{-\frac{U(x)}{kT_{0}}}
    The probability at excited thermodynamic states is:
    .. math:
        \pi^I(x) = \mathrm{e}^{-\frac{U(x)}{kT_{I}}}
                 = \mathrm{e}^{-\frac{U(x)}{kT_{0}}+\frac{U(x)}{kT_{0}}-\frac{U(x)}{kT_{I}}}
                 = \mathrm{e}^{-\frac{U(x)}{kT_{0}}}\mathrm{e}^{-\left(\frac{U(x)}{kT_{I}}-\frac{U(x)}{kT_{0}}\right)}
                 = \mathrm{e}^{-u(x)}\mathrm{e}^{-b_{I}(x)}
    where we have defined the bias energies:
    .. math:
        b_{I}(x) = U(x)\left(\frac{1}{kT_{I}}-\frac{1}{kT_{0}}\right)
    Parameters
    ----------
    utrajs : ndarray or list of ndarray
        Potential energy trajectories.
    ttrajs : ndarray or list of ndarray
        Generating thermodynamic state trajectories.
    kTs : ndarray of float
        kT values of the different temperatures.
    """
    pass

# ===================================
# Estimators
# ===================================

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
    maxiter : int, optional, default=10000
        The maximum number of dTRAM iterations before the estimator exits unsuccessfully.
    maxerr : float, optional, default=1e-15
        Convergence criterion based on the maximal free energy change in a self-consistent
        iteration step.

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
    from pyemma.thermo.estimators import DTRAM
    dtram_estimator = DTRAM(
        bias, lag=lag, count_mode='sliding',
        maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info,
        dt_traj=dt_traj, init=init)
    # run estimation
    return dtram_estimator.estimate(X)

def wham(ttrajs, dtrajs, bias, maxiter=100000, maxerr=1.0E-15, save_convergence_info=0):
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

    Returns
    -------
    ??? : ???
        A ??? model which consists of stationary quantities at all
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
    # build WHAM
    from pyemma.thermo.estimators import WHAM
    wham_estimator = WHAM(
        bias, maxiter=maxiter, maxerr=maxerr, save_convergence_info=save_convergence_info)
    # run estimation
    return wham_estimator.estimate(X)
