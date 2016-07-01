
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

__author__ = 'noe'


def contour(x, y, z, ncontours = 50, colorbar=True, fig=None, ax=None, method='linear', zlim=None, cmap=None):
    import matplotlib.pylab as _plt
    from scipy.interpolate import griddata as gd
    # check input
    if ax is None:
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


def plot_free_energy(xall, yall, weights=None, ax=None, nbins=100, ncountours=100, offset=-1,
                     avoid_zero_count=True, minener_zero=True, kT=1.0, vmin=0.0, vmax=None,
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
    ncountours : int, default=100
        number of contours used
    offset : float, default=0.1
        DEPRECATED and ineffective.
    avoid_zero_count : bool, default=True
        avoid zero counts by lifting all histogram elements to the minimum value
        before computing the free energy. If False, zero histogram counts will
        yield NaNs in the free energy which and thus regions that are not plotted.
    minener_zero : bool, default=True
        Shifts the energy minimum to zero. If false, will not shift at all.
    kT : float, default=1.0
        The value of kT in the desired energy unit. By default, will compute
        energies in kT (setting 1.0). If you want to measure the energy in
        kJ/mol at 298 K, use kT=2.479 and change the cbar_label accordingly.
    vmin : float or None, default=0.0
        Lowest energy that will be plotted
    vmax : float or None, default=None
        Highest energy that will be plotted
    cmap : matplotlib colormap, optional, default = None
        the color map to use. None will use pylab.cm.spectral.
    cbar : boolean, default=True
        plot a color bar
    cbar_label : str or None, default='Free energy (kT)'
        colorbar label string. Use None to suppress it.

    Returns
    -------
    fig : Figure object containing the plot

    ax : Axes object containing the plot

    """
    import matplotlib.pylab as _plt
    import warnings

    # check input
    if offset != -1:
        warnings.warn("Parameter offset is deprecated and will be ignored", DeprecationWarning)
    # histogram
    z, xedge, yedge = _np.histogram2d(xall, yall, bins=nbins, weights=weights)
    x = 0.5*(xedge[:-1] + xedge[1:])
    y = 0.5*(yedge[:-1] + yedge[1:])
    # avoid zeros
    if avoid_zero_count:
        zmin_nonzero = _np.min(z[_np.where(z > 0)])
        z = _np.maximum(z, zmin_nonzero)
    # compute free energies
    F = -kT * _np.log(z)
    if minener_zero:
        F -= _np.min(F)
    # do a contour plot
    extent = [yedge[0], yedge[-1], xedge[0], xedge[-1]]
    if ax is None:
        ax = _plt.gca()
    CS = ax.contourf(x, y, F.T, ncountours, extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    if cbar:
        cbar = _plt.colorbar(CS)
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label)

    return _plt.gcf(), ax

