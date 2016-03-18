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
        if isinstance(obj, _WHAM):
            label = "WHAM"
        elif isinstance(obj, _DTRAM):
            label = "dTRAM, lag=%d" % obj.lag
        else:
            label = None
        ax.plot(
            (_np.arange(obj.increments.shape[0]) + 1) * obj.save_convergence_info,
            obj.increments, '-s', label=label)
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
        if isinstance(obj, _WHAM):
            label = "WHAM"
        elif isinstance(obj, _DTRAM):
            label = "dTRAM, lag=%d" % obj.lag
        else:
            label = None
        ax.plot(
            (_np.arange(1, obj.loglikelihoods.shape[0]) + 1) * obj.save_convergence_info,
            obj.loglikelihoods[1:] - obj.loglikelihoods[:-1], '-s', label=label)
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



