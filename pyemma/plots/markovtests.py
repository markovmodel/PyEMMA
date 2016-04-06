
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from __future__ import absolute_import
__author__ = 'noe'

import math
import numpy as np


def _add_ck_subplot(cktest, ax, i, j, ipos=None, jpos=None, y01=True, units='steps', dt=1.):
    # plot estimates
    lest = ax.plot(dt*cktest.lagtimes, cktest.estimates[:, i, j], color='black')
    # plot error of estimates if available
    if cktest.has_errors and cktest.err_est:
        ax.fill_between(dt*cktest.lagtimes, cktest.estimates_conf[0][:, i, j], cktest.estimates_conf[1][:, i, j],
                        color='black', alpha=0.2)
    # plot predictions
    lpred = ax.plot(dt*cktest.lagtimes, cktest.predictions[:, i, j], color='blue', linestyle='dashed')
    # plot error of predictions if available
    if cktest.has_errors:
        ax.fill_between(dt*cktest.lagtimes, cktest.predictions_conf[0][:, i, j], cktest.predictions_conf[1][:, i, j],
                        color='blue', alpha=0.2)
    # add label
    ax.text(0.05, 0.05, str(i+1)+' -> '+str(j+1), transform=ax.transAxes, weight='bold')
    if y01:
        ax.set_ylim(0, 1)
    # Axes labels
    if ipos is None:
        ipos = i
    if jpos is None:
        jpos = j
    if (jpos == 0):
        ax.set_ylabel('probability')
    if (ipos == cktest.nsets-1):
        ax.set_xlabel('lag time (' + units + ')')
    # return line objects
    return lest, lpred


def plot_cktest(cktest, figsize=None, diag=False,  y01=True, layout=None,
                padding_between=0.1, padding_top=0.075, units='steps', dt=1.):
    """Plot of Chapman-Kolmogorov test

    Parameters
    ----------

    cktest : msm.ChapmanKolmogorovValidator
        Chapman-Kolmogorov Test

    figsize : shape, default=(10, 10)
        Figure size

    diag : bool, default=False
        Plot only diagonal elements of the test, i.e. self-transition
        probabilities.

    y01 : bool, default=True
        Scale all y-Axes to [0,1]. If True, the y-Axes can be shared
        and the figure is tighter. If False, each y Axis will be scaled
        automatically.

    layout : str or shape or None, default=None
        Organization of subplots. You can specify your own shape. If None,
        an automatic shape will be selected. Use 'wide' for plots that
        span the page width (double-column figures) and 'tall' for
        single-column figures.

    padding_between : float, default=0.1
        padding space between subplots (as a fraction of 1.0)

    padding_top : float, default=0.05
        padding space on top of subplots (as a fraction of 1.0)

    Returns
    -------
    fig : Figure object

    axes : Axis objects with subplots

    """
    import matplotlib.pylab as plt

    sharey = y01
    # first fix subfigure layout
    if diag:
        if layout is None or layout == 'wide':
            ncol = min(4, cktest.nsets)
            layout = (int(math.ceil(cktest.nsets / ncol)), ncol)
        elif layout == 'tall':
            nrow = min(4, cktest.nsets)
            layout = (nrow, int(math.ceil(cktest.nsets / nrow)))
    else:
        layout = (cktest.nsets, cktest.nsets)
    # fix figure size
    if figsize is None:
        size_per_subplot = min(3.0, 10.0 / np.max(np.array(layout)))
        figsize = (size_per_subplot*layout[1], size_per_subplot*layout[0])
    # generate subplots
    fig, axes = plt.subplots(layout[0], layout[1], sharex=True, sharey=sharey, figsize=figsize)
    axeslist = list(axes.flatten())
    # line objects
    lest = None
    lpred = None
    # plot
    for (k, ax) in enumerate(axeslist):
        if diag and k < cktest.nsets:
            ipos = int(k/layout[1])
            jpos = int(k%layout[1])
            lest, lpred = _add_ck_subplot(cktest, ax, k, k, ipos=ipos, jpos=jpos, y01=y01, units=units, dt=dt)
            k += 1
        else:
            i = int(k/cktest.nsets)
            j = int(k%cktest.nsets)
            lest, lpred = _add_ck_subplot(cktest, ax, i, j, y01=y01, units=units, dt=dt)
    # figure legend
    predlabel = 'predict'
    estlabel = 'estimate'
    if cktest.has_errors:
        predlabel += '     conf. {:3.1f}%'.format(100.0*cktest.conf)
    fig.legend((lest[0], lpred[0]), (estlabel, predlabel), 'upper center', ncol=2, frameon=False)
    # change subplot padding
    plt.subplots_adjust(top=1.0-padding_top, wspace=padding_between, hspace=padding_between)
    # done
    return fig, axes
