
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

import numpy as _np
from scipy.interpolate import griddata as gd

__author__ = 'noe'


def contour(x, y, z, ncontours = 50, colorbar=True, fig=None, ax=None, method='linear', zlim=None, cmap=None):
    import matplotlib.pylab as _plt
    # check input
    if (ax is None):
        if fig is None:
            ax = _plt.gca()
        else:
            ax = fig.gca()

    # grid data
    points = _np.hstack([x[:,None],y[:,None]])
    xi, yi = _np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
    zi = gd(points, z, (xi, yi), method=method)
    # contour level levels
    if zlim is None:
        zlim = (z.min(), z.max())
    eps = (zlim[1] - zlim[0]) / float(ncontours)
    levels = _np.linspace(zlim[0] - eps, zlim[1] + eps)
    # contour plot
    if cmap is None:
        cmap=_plt.cm.jet
    cf = ax.contourf(xi, yi, zi, ncontours, cmap=cmap, levels=levels)
    # color bar if requested
    if colorbar:
        _plt.colorbar(cf, ax=ax)

    return ax


def scatter_contour(x, y, z, ncontours = 50, colorbar=True, fig=None, ax=None, cmap=None, outfile=None):
    """Contour plot on scattered data (x,y,z) and the plots the positions of the points (x,y) on top.

    Parameters
    ----------
    x : ndarray(T)
        x-coordinates
    y : ndarray(T)
        y-coordinates
    z : ndarray(T)
        z-coordinates
    ncontours : int, optional, default = 50
        number of contour levels
    fig : matplotlib Figure object, optional, default = None
        the figure to plot into. When set to None the default Figure object will be used
    ax : matplotlib Axes object, optional, default = None
        the axes to plot to. When set to None the default Axes object will be used.
    cmap : matplotlib colormap, optional, default = None
        the color map to use. None will use pylab.cm.jet.
    outfile : str, optional, default = None
        output file to write the figure to. When not given, the plot will be displayed

    Returns
    -------
    ax : Axes object containing the plot

    """
    import matplotlib.pylab as _plt
    ax = contour(x, y, z, ncontours=ncontours, colorbar=colorbar, fig=fig, ax=ax, cmap=cmap)

    # scatter points
    ax.scatter(x,y,marker='o',c='b',s=5)

    # show or save
    if outfile is not None:
        _plt.savefig(outfile)

    return ax


def plot_free_energy(xall, yall, weights=None, ax=None, nbins=100, offset=0.1,
                     cmap='spectral', cbar=True, cbar_label='Free energy (kT)'):
    """Free energy plot given 2D scattered data

    Builds a 2D-histogram of the given data points and plots -log(p) where p is
    the probability computed from the histogram count.

    Parameters
    ----------
    xall : ndarray(T)
        sample x-coordinates
    yall : ndarray(T)
        sample y-coordinates
    weights : ndarray(T), default = None
        sample weights. By default all samples have the same weight
    ax : matplotlib Axes object, default = None
        the axes to plot to. When set to None the default Axes object will be used.
    nbins : int, default=100
        number of histogram bins used in each dimension
    offset : float, default=0.1
        small additive shift to the histogram. This creates a small bias to the
        distribution, but gives a better visual impression with the default
        colormap.
    cmap : matplotlib colormap, optional, default = None
        the color map to use. None will use pylab.cm.spectral.
    cbar : boolean, default=True
        plot a color bar
    cbar_label : str or None, default='Free energy (kT)'
        colorbar label string. Use None to suppress it.

    Returns
    -------
    ax : Axes object containing the plot

    fig : Figure object containing the plot

    """
    import matplotlib.pylab as _plt

    z, x, y = _np.histogram2d(xall, yall, bins=nbins, weights=weights)
    z += offset
    # compute free energies
    F = -_np.log(z)
    # do a contour plot
    extent = [x[0], x[-1], y[0], y[-1]]
    if ax is None:
        ax = _plt.gca()
    CS = ax.contourf(F.T, 100, extent=extent, cmap=cmap)
    if cbar:
        cbar = _plt.colorbar(CS)
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label)

    return ax, _plt.gcf()

