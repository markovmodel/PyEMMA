
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group
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
from six.moves import range
__author__ = 'noe'

import numpy as _np
import matplotlib.pylab as _plt

def plot_implied_timescales(ITS, ax=None, outfile=None, show_mle=True, show_mean=True,
                            xlog=False, ylog=True, confidence=0.95, refs=None,
                            units='steps', dt=1., **kwargs):
    r""" Generate a pretty implied timescale plot

    Parameters
    ----------
    ITS : implied timescales object.
        Object whose data will be plotted. Must provide the functions: get_timescales() and get_timescales(i) where i is the
        the property samples_available
    ax : matplotlib Axes object, optional, default = None
        the axes to plot to. When set to None the default Axes object will be used.
    outfile : str, optional, default = None
        output file to write the figure to. When not given, the plot will be displayed
    show_mean : bool, default = True
        Line for mean value will be shown, if available
    show_mle : bool, default = True
        Line for maximum likelihood estimate will be shown
    xlog : bool, optional, default = False
        Iff true, the x-Axis is logarithmized
    ylog : bool, optional, default = True
        Iff true, the y-Axis is logarithmized
    confidence : float, optional, default = 0.95
        The confidence interval for plotting error bars (if available)
    refs : ndarray((m), dtype=float), optional, default = None
        Reference (exact solution or other reference) timescales if known. The number of timescales must match those
        in the ITS object
    units: str, optional, default = 'steps'
        Affects the labeling of the axes. Used with :py:obj:`dt`, allows for changing the physical units of the axes.
        Accepts simple LaTeX math strings, eg. '$\mu$s'
    dt: float, optional, default = 1.0
        Physical time between frames, expressed the units given in :py:obj:`units`. E.g, if you know that each
        frame corresponds to .010 ns, you can use the combination of parameters :py:obj:`dt` =0.01,
        :py:obj:`units` ='ns' to display the implied timescales in ns (instead of frames)

    **kwargs: Will be parsed to pyplot.plo when plotting the MLE datapoints (not the bootstrapped means).
            See the doc of pyplot for more options. Most useful lineproperties like `marker='o'` and/or :markersize=5

    Returns
    -------
    ax : Axes object containing the plot

    """
    # check input
    if (ax is None):
        ax = _plt.gca()
    colors = ['blue','red','green','cyan','purple','orange','violet']
    lags = ITS.lagtimes
    xmax = _np.max(lags)
    #ymin = min(_np.min(lags), _np.min(ITS.get_timescales()))
    #ymax = 1.5*_np.min(ITS.get_timescales())
    for i in range(ITS.number_of_timescales):
        # plot estimate
        if show_mle:
            ax.plot(lags*dt, ITS.get_timescales(process=i)*dt, color=colors[i % len(colors)], **kwargs)
        # sample available?
        if (ITS.samples_available):# and ITS.sample_number_of_timescales > i):
            # plot sample mean
            if show_mean:
                ax.plot(lags*dt, ITS.get_sample_mean(process=i)*dt, marker='o',
                        color=colors[i % len(colors)], linestyle='dashed')
            (lconf, rconf) = ITS.get_sample_conf(confidence, i)
            ax.fill_between(lags*dt, lconf*dt, rconf*dt, alpha=0.2, color=colors[i % len(colors)])
        # reference available?
        if (refs is not None):
            tref = refs[i]
            ax.plot([0,min(tref,xmax)]*dt, [tref,tref]*dt, color='black', linewidth=1)
    # cutoff
    ax.plot(lags*dt, lags*dt, linewidth=2, color='black')
    ax.set_xlim([1*dt,xmax*dt])
    #ax.set_ylim([ymin,ymax])
    ax.fill_between(lags*dt, ax.get_ylim()[0]*_np.ones(len(lags))*dt, lags*dt, alpha=0.5, color='grey')
    # formatting
    ax.set_xlabel('lag time / %s'%units)
    ax.set_ylabel('timescale / %s'%units)
    if (xlog):
        ax.set_xscale('log')
    if (ylog):
        ax.set_yscale('log')

    # show or save
    # if outfile is None:
    #    _plt.show()
    if outfile is not None:
        _plt.savefig(outfile)

    return ax