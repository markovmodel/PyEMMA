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

def plot_convergence_info(thermo_estimator, figsize=(12, 4.5)):
    from pyemma.thermo import WHAM as _WHAM
    from pyemma.thermo import DTRAM as _DTRAM
    fig, ax = _plt.subplots(1, 2, figsize=figsize)
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
        ax[0].plot(
            (_np.arange(obj.increments.shape[0]) + 1) * obj.save_convergence_info,
            obj.increments, '-s', markersize=10, label=label)
        ax[1].plot(
            (_np.arange(1, obj.loglikelihoods.shape[0]) + 1) * obj.save_convergence_info,
            obj.loglikelihoods[1:] - obj.loglikelihoods[:-1], '-s', markersize=10, label=label)

    ax[0].set_ylabel(r"increment / kT", fontsize=20)
    ax[1].set_ylabel(r"loglikelihood increase", fontsize=20)
    for _ax in ax:
        _ax.set_xlabel(r"iteration", fontsize=20)
        _ax.tick_params(labelsize=15)
        _ax.semilogx()
        _ax.semilogy()
        _ax.legend(loc=1, fontsize=12, fancybox=True, framealpha=0.5)
    fig.tight_layout()
    return fig, ax



