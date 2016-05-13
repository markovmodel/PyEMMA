# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
import matplotlib.pyplot as _plt
from pyemma.thermo import WHAM as _WHAM
from pyemma.thermo import DTRAM as _DTRAM
from pyemma.thermo import TRAM as _TRAM
from pyemma.msm import MSM as _MSM

__all__ = [
    'plot_increments',
    'plot_loglikelihoods',
    'plot_convergence_info',
    'plot_memm_implied_timescales']

def get_estimator_label(thermo_estimator):
    if isinstance(thermo_estimator, _WHAM):
        return "WHAM"
    elif isinstance(thermo_estimator, _DTRAM):
        return "dTRAM, lag=%d" % thermo_estimator.lag
    elif isinstance(thermo_estimator, _TRAM):
        return "TRAM, lag=%d" % thermo_estimator.lag
    return None

def plot_increments(thermo_estimator, ax=None):
    # TODO: write docstring
    if ax is None:
        fig, ax = _plt.subplots()
    else:
        fig = None
    obj_list = thermo_estimator
    if not isinstance(obj_list, (list, tuple)):
        obj_list = [obj_list]
    for obj in obj_list:
        ax.plot(
            (_np.arange(obj.increments.shape[0]) + 1) * obj.save_convergence_info,
            obj.increments, '-s', label=get_estimator_label(obj))
    ax.set_xlabel(r"iteration")
    ax.set_ylabel(r"increment / kT")
    ax.semilogx()
    ax.semilogy()
    ax.legend(loc=1, fancybox=True, framealpha=0.5)
    return ax

def plot_loglikelihoods(thermo_estimator, ax=None):
    # TODO: write docstring
    if ax is None:
        fig, ax = _plt.subplots()
    else:
        fig = None
    obj_list = thermo_estimator
    if not isinstance(obj_list, (list, tuple)):
        obj_list = [obj_list]
    for obj in obj_list:
        ax.plot(
            (_np.arange(1, obj.loglikelihoods.shape[0]) + 1) * obj.save_convergence_info,
            obj.loglikelihoods[1:] - obj.loglikelihoods[:-1], '-s', label=get_estimator_label(obj))
    ax.set_xlabel(r"iteration")
    ax.set_ylabel(r"loglikelihood increase")
    ax.semilogx()
    ax.semilogy()
    ax.legend(loc=1, fancybox=True, framealpha=0.5)
    return ax

def plot_convergence_info(thermo_estimator, axes=None):
    # TODO: write docstring
    if axes is None:
        fs = _plt.rcParams['figure.figsize']
        fig, axes = _plt.subplots(2, 1, figsize=(fs[0], fs[1] * 1.5), sharex=True)
    else:
        assert len(axes) == 2
        fig = None
    plot_increments(thermo_estimator, ax=axes[0])
    plot_loglikelihoods(thermo_estimator, ax=axes[1])
    if fig is not None:
        fig.tight_layout()
        axes[0].set_xlabel('')
    return axes

def plot_memm_implied_timescales(thermo_estimators,
    ax=None, nits=None, therm_state=None, xlog=False, ylog=True, units='steps', dt=1.0, refs=None,
    annotate=True, **kwargs):
    colors = ['blue', 'red', 'green', 'cyan', 'purple', 'orange', 'violet']
    # Check units and dt for user error.
    if isinstance(units, list) and len(units) != 2:
        raise TypeError("If units is a list, len(units) has to be = 2")
    if isinstance(dt, list) and len(dt) != 2:
        raise TypeError("If dt is a list, len(dt) has to be = 2")
    # Create list of units and dts for different axis
    if isinstance(units, str):
        units = [units] * 2
    if isinstance(dt, (float, int)):
        dt = [dt] * 2
    # Create figure/ax
    if ax is None:
        fig, ax = _plt.subplots()
    else:
        fig = None
    # Get timescales from all MEMM instances for the requested thermodynamic state
    lags = []
    ts = []
    for memm in thermo_estimators:
        assert isinstance(memm, (_DTRAM, _TRAM)), 'only dTRAM + TRAM accepted'
        if therm_state is None:
            assert memm.msm is not None, 'unbiased observations required'
            msm = memm.msm
        else:
            assert 0 <= therm_state < memm.nthermo, 'therm_state out of range'
            msm = memm.models[therm_state]
        assert isinstance(msm, _MSM), 'kinetic model must be MSM'
        lags.append(memm.lag)
        ts.append(msm.timescales(k=nits))
    lags = _np.asarray(lags)
    ts = _np.asarray(ts)
    srt = _np.argsort(lags)
    # Plot the implied timescales
    for i in range(ts.shape[1]):
        ax.plot(lags[srt], ts[srt, i], color=colors[i % len(colors)], **kwargs)
    # Set boundaries
    ax.set_xlim([lags.min() * dt[0], lags.max() * dt[0]])
    # Plot cutoff
    ax.plot(lags[srt] * dt[0], lags[srt] * dt[1], linewidth=2, color='black')
    ax.fill_between(
        lags[srt] * dt[0], _np.max([1, ax.get_ylim()[0]]) * _np.ones(len(lags)) * dt[1],
        lags[srt] * dt[1], alpha=0.5, color='grey')
    # formatting
    ax.set_xlabel('lag time / %s'%units[0])
    ax.set_ylabel('timescale / %s'%units[1])
    # Make the plot look pretty
    if xlog:
        ax.semilogx()
    if ylog:
        ax.semilogy()
    if therm_state is None:
        state_label = "unbiased"
    else:
        state_label = "therm_state=%d" % therm_state
    if annotate:
        ax.text(0.02, 0.95, state_label, transform=ax.transAxes)
    return ax
