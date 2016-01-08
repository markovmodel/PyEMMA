# Copyright (c) 2015 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as _np
from pyemma.util import types as _types

__docformat__ = "restructuredtext en"
__author__ = "Frank Noe, Christoph Wehmeyer"
__copyright__ = "Copyright 2015, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Frank Noe", "Christoph Wehmeyer"]
__maintainer__ = "Christoph Wehmeyer"
__email__ = "christoph.wehmeyer@fu-berlin.de"

__all__ = ['dtram', 'wham']

# ===================================
# Data Loaders and Readers
# ===================================

# TODO: what about simple molecular dynamics data? How do we combine MD data with US data?

# This corresponds to the source function in coordinates.api
def umbrella_sampling_data(umbrella_trajs, centers, k, md_trajs=None, nbin=None):
    r"""
    Wraps umbrella sampling data or a mix of umbrella sampling and and direct molecular dynamics

    Parameters
    ----------
    umbrella_trajs : list of K arrays, each of shape (T_i, d)
        List of arrays, each having T_i rows, one for each time step, and d columns where d is the
        dimension in which umbrella sampling was applied. Often d=1, and thus umbrella_trajs will
        be a list of 1d-arrays.
    centers : array-like of size K
        List or array of K center positions. Each position must be a d-dimensional vector. For 1d
        umbrella sampling, one can simply pass a list of centers, e.g. [-5.0, -4.0, -3.0, ... ].
    k : int or array-like of int
        The force constant used in the umbrellas, unit-less (e.g. kT per length unit). If different
        force constants were used for different umbrellas, a list or array of K force constants
        can be given. For multidimensional umbrella sampling, the force matrix must be used.
    md_trajs : list of K arrays, each of shape (T_i, d), optional, default=None
        Unbiased molecular dynamics simulations. Format like umbrella_trajs.
    nbin : int
        ???
    """
    pass

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

def dtram(ttrajs, dtrajs, bias, lag, maxiter=10000, maxerr=1.0E-15, err_out=0, lll_out=0, dt_traj='1 step'):
    r"""
    Discrete transition-based reweighting analysis method
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
    dtram_estimator = DTRAM(bias, lag=lag, count_mode='sliding',
        maxiter=maxiter, maxerr=maxerr, err_out=err_out, lll_out=lll_out, dt_traj=dt_traj)
    # run estimation
    return dtram_estimator.estimate(X)

def wham(ttrajs, dtrajs, bias, maxiter=100000, maxerr=1.0E-15, err_out=0, lll_out=0):
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
        bias, maxiter=maxiter, maxerr=maxerr, err_out=err_out, lll_out=lll_out)
    # run estimation
    return wham_estimator.estimate(X)
