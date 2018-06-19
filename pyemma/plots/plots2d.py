
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2018 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
from warnings import warn as _warn

__author__ = 'noe'


def _get_cmap(cmap):
    # matplotlib 2.0 deprecated 'spectral' colormap, renamed to nipy_spectral.
    from matplotlib import __version__
    version = tuple(map(int, __version__.split('.')))
    if cmap == 'spectral' and version >= (2, ):
        cmap = 'nipy_spectral'
    return cmap


def contour(
        x, y, z, ncontours=50, colorbar=True, fig=None, ax=None,
        method='linear', zlim=None, cmap=None):
    import matplotlib.pyplot as _plt
    _warn(
        'contour is deprected; use plot_contour instead.',
        DeprecationWarning)
    cmap = _get_cmap(cmap)
    if cmap is None:
        cmap = 'jet'
    if ax is None:
        if fig is None:
            fig, ax = _plt.subplots()
        else:
            ax = fig.gca()
    if zlim is None:
        vmin, vmax = None, None
    else:
        vmin, vmax = zlim
    _, ax, _ = plot_contour(
        x, y, z, ax=ax, cmap=cmap,
        ncontours=ncontours, vmin=vmin, vmax=vmax, levels=None,
        cbar=colorbar, cax=None, cbar_label=None, logscale=False,
        nbins=100, method=method)
    return ax


def scatter_contour(
        x, y, z, ncontours=50, colorbar=True, fig=None,
        ax=None, cmap=None, outfile=None):
    """Contour plot on scattered data (x,y,z) and
    plots the positions of the points (x,y) on top.

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
        the figure to plot into. When set to None the default
        Figure object will be used
    ax : matplotlib Axes object, optional, default = None
        the axes to plot to. When set to None the default Axes
        object will be used.
    cmap : matplotlib colormap, optional, default = None
        the color map to use. None will use pylab.cm.jet.
    outfile : str, optional, default = None
        output file to write the figure to. When not given,
        the plot will be displayed

    Returns
    -------
    ax : Axes object containing the plot

    """
    _warn(
        'scatter_contour is deprected; use plot_contour instead'
        ' and manually add a scatter plot on top.',
        DeprecationWarning)
    ax = contour(
        x, y, z, ncontours=ncontours, colorbar=colorbar,
        fig=fig, ax=ax, cmap=cmap)
    # scatter points
    ax.scatter(x , y, marker='o', c='b', s=5)
    # show or save
    if outfile is not None:
        ax.get_figure().savefig(outfile)
    return ax


# ######################################################################
#
# auxiliary functions to help constructing custom plots
#
# ######################################################################


def get_histogram(
        xall, yall, nbins=100,
        weights=None, avoid_zero_count=False):
    """Compute a two-dimensional histogram.

    Parameters
    ----------
    xall : ndarray(T)
        Sample x-coordinates.
    yall : ndarray(T)
        Sample y-coordinates.
    nbins : int, optional, default=100
        Number of histogram bins used in each dimension.
    weights : ndarray(T), optional, default=None
        Sample weights; by default all samples have the same weight.
    avoid_zero_count : bool, optional, default=True
        Avoid zero counts by lifting all histogram elements to the
        minimum value before computing the free energy. If False,
        zero histogram counts would yield infinity in the free energy.

    Returns
    -------
    x : ndarray(nbins, nbins)
        The bins' x-coordinates in meshgrid format.
    y : ndarray(nbins, nbins)
        The bins' y-coordinates in meshgrid format.
    z : ndarray(nbins, nbins)
        Histogram counts in meshgrid format.

    """
    z, xedge, yedge = _np.histogram2d(
        xall, yall, bins=nbins, weights=weights)
    x = 0.5 * (xedge[:-1] + xedge[1:])
    y = 0.5 * (yedge[:-1] + yedge[1:])
    if avoid_zero_count:
        z = _np.maximum(z, _np.min(z[z.nonzero()]))
    return x, y, z


def get_grid_data(xall, yall, zall, nbins=100, method='nearest'):
    """Interpolate unstructured two-dimensional data.

    Parameters
    ----------
    xall : ndarray(T)
        Sample x-coordinates.
    yall : ndarray(T)
        Sample y-coordinates.
    zall : ndarray(T)
        Sample z-coordinates.
    nbins : int, optional, default=100
        Number of histogram bins used in x/y-dimensions.
    method : str, optional, default='nearest'
        Assignment method; scipy.interpolate.griddata supports the
        methods 'nearest', 'linear', and 'cubic'.

    Returns
    -------
    x : ndarray(nbins, nbins)
        The bins' x-coordinates in meshgrid format.
    y : ndarray(nbins, nbins)
        The bins' y-coordinates in meshgrid format.
    z : ndarray(nbins, nbins)
        Interpolated z-data in meshgrid format.

    """
    from scipy.interpolate import griddata
    x, y = _np.meshgrid(
        _np.linspace(xall.min(), xall.max(), nbins),
        _np.linspace(yall.min(), yall.max(), nbins),
        indexing='ij')
    z = griddata(
        _np.hstack([xall[:,None], yall[:,None]]),
        zall, (x, y), method=method)
    return x, y, z


def _to_density(z):
    """Normalize histogram counts.

    Parameters
    ----------
    z : ndarray(T)
        Histogram counts.

    """
    return z / float(z.sum())


def _to_free_energy(z, minener_zero=False):
    """Compute free energies from histogram counts.

    Parameters
    ----------
    z : ndarray(T)
        Histogram counts.
    minener_zero : boolean, optional, default=False
        Shifts the energy minimum to zero.

    Returns
    -------
    free_energy : ndarray(T)
        The free energy values in units of kT.

    """
    pi = _to_density(z)
    free_energy = _np.inf * _np.ones(shape=z.shape)
    nonzero = pi.nonzero()
    free_energy[nonzero] = -_np.log(pi[nonzero])
    if minener_zero:
        free_energy[nonzero] -= _np.min(free_energy[nonzero])
    return free_energy


def plot_map(
        x, y, z, ax=None, cmap='Blues',
        ncontours=100, vmin=None, vmax=None, levels=None,
        cbar=True, cax=None, cbar_label=None, logscale=False):
    """Plot a two-dimensional map from data on a grid.

    Parameters
    ----------
    x : ndarray(T)
        Binned x-coordinates.
    y : ndarray(T)
        Binned y-coordinates.
    z : ndarray(T)
        Binned z-coordinates.
    ax : matplotlib.Axes object, optional, default=None
        The ax to plot to; if ax=None, a new ax (and fig) is created.
    cmap : matplotlib colormap, optional, default='Blues'
        The color map to use.
    ncontours : int, optional, default=100
        Number of contour levels.
    vmin : float, optional, default=None
        Lowest z-value to be plotted.
    vmax : float, optional, default=None
        Highest z-value to be plotted.
    levels : iterable of float, optional, default=None
        Contour levels to plot.
    cbar : boolean, optional, default=True
        Plot a color bar.
    cax : matplotlib.Axes object, optional, default=None
        Plot the colorbar into a custom axes object instead of
        stealing space from ax.
    cbar_label : str, optional, default=None
        Colorbar label string; use None to suppress it.
    logscale : boolean, optional, default=False
        Plot the z-values in logscale.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    cbar : matplotlib.Colorbar object
        The corresponding colorbar object; None if no colorbar
        was requested.

    """
    import matplotlib.pyplot as _plt
    if ax is None:
        fig, ax = _plt.subplots()
    else:
        fig = ax.get_figure()
    if logscale:
        from matplotlib.colors import LogNorm
        norm = LogNorm()
        z = _np.ma.masked_where(z <= 0, z)
    else:
        norm = None
    cs = ax.contourf(
        x, y, z, ncontours, norm=norm,
        vmin=vmin, vmax=vmax, cmap=cmap,
        levels=levels)
    if cbar:
        if cax is None:
            cbar = fig.colorbar(cs, ax=ax)
        else:
            cbar = fig.colorbar(cs, cax=cax)
        if cbar_label is not None:
            cbar.set_label(cbar_label)
    else:
        cbar = None
    return fig, ax, cbar


# ######################################################################
#
# new plotting functions
#
# ######################################################################


def plot_density(
        xall, yall, ax=None, cmap='Blues',
        ncontours=100, vmin=None, vmax=None, levels=None,
        cbar=True, cax=None, cbar_label=None, logscale=False,
        nbins=100, weights=None, avoid_zero_count=False,):
    """Plot a two-dimensional density map using a histogram of
    scattered data.

    Parameters
    ----------
    xall : ndarray(T)
        Sample x-coordinates.
    yall : ndarray(T)
        Sample y-coordinates.
    ax : matplotlib.Axes object, optional, default=None
        The ax to plot to; if ax=None, a new ax (and fig) is created.
    cmap : matplotlib colormap, optional, default='Blues'
        The color map to use.
    ncontours : int, optional, default=100
        Number of contour levels.
    vmin : float, optional, default=None
        Lowest z-value to be plotted.
    vmax : float, optional, default=None
        Highest z-value to be plotted.
    levels : iterable of float, optional, default=None
        Contour levels to plot.
    cbar : boolean, optional, default=True
        Plot a color bar.
    cax : matplotlib.Axes object, optional, default=None
        Plot the colorbar into a custom axes object instead of
        stealing space from ax.
    cbar_label : str, optional, default=None
        Colorbar label string; use None to suppress it.
    logscale : boolean, optional, default=False
        Plot the z-values in logscale.
    nbins : int, optional, default=100
        Number of histogram bins used in each dimension.
    weights : ndarray(T), optional, default=None
        Sample weights; by default all samples have the same weight.
    avoid_zero_count : bool, optional, default=True
        Avoid zero counts by lifting all histogram elements to the
        minimum value before computing the free energy. If False,
        zero histogram counts would yield infinity in the free energy.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    cbar : matplotlib.Colorbar object
        The corresponding colorbar object; None if no colorbar
        was requested.

    """
    x, y, z = get_histogram(
        xall, yall, nbins=nbins, weights=weights,
        avoid_zero_count=avoid_zero_count)
    return plot_map(
        x, y, _to_density(z).T, ax=ax, cmap=cmap,
        ncontours=ncontours, vmin=vmin, vmax=vmax, levels=levels,
        cbar=cbar, cax=cax, cbar_label=cbar_label, logscale=logscale)


def plot_free_energy(
        xall, yall, weights=None, ax=None, nbins=100, ncontours=100,
        offset=-1, avoid_zero_count=False, minener_zero=True, kT=1.0,
        vmin=None, vmax=None, cmap='nipy_spectral', cbar=True,
        cbar_label='free energy / kT', cax=None, levels=None,
        logscale=False, legacy=True, ncountours=None):
    """Plot a two-dimensional free energy map using a histogram of
    scattered data.

    Parameters
    ----------
    xall : ndarray(T)
        Sample x-coordinates.
    yall : ndarray(T)
        Sample y-coordinates.
    weights : ndarray(T), optional, default=None
        Sample weights; by default all samples have the same weight.
    ax : matplotlib.Axes object, optional, default=None
        The ax to plot to; if ax=None, a new ax (and fig) is created.
        Number of contour levels.
    nbins : int, optional, default=100
        Number of histogram bins used in each dimension.
    ncontours : int, optional, default=100
        Number of contour levels.
    offset : float, optional, default=-1
        Deprecated and ineffective; raises a ValueError
        outside legacy mode.
    avoid_zero_count : bool, optional, default=False
        Avoid zero counts by lifting all histogram elements to the
        minimum value before computing the free energy. If False,
        zero histogram counts would yield infinity in the free energy.
    minener_zero : boolean, optional, default=True
        Shifts the energy minimum to zero.
    kT : float, optional, default=1.0
        The value of kT in the desired energy unit. By default,
        energies are computed in kT (setting 1.0). If you want to
        measure the energy in kJ/mol at 298 K, use kT=2.479 and
        change the cbar_label accordingly.
    vmin : float, optional, default=None
        Lowest free energy value to be plotted.
        (default=0.0 in legacy mode)
    vmax : float, optional, default=None
        Highest free energy value to be plotted.
    cmap : matplotlib colormap, optional, default='nipy_spectral'
        The color map to use.
    cbar : boolean, optional, default=True
        Plot a color bar.
    cbar_label : str, optional, default='free energy / kT'
        Colorbar label string; use None to suppress it.
    cax : matplotlib.Axes object, optional, default=None
        Plot the colorbar into a custom axes object instead of
        stealing space from ax.
    levels : iterable of float, optional, default=None
        Contour levels to plot.
    logscale : boolean, optional, default=False
        Plot the z-values in logscale.
    legacy : boolean, optional, default=True
        Switch to use the function in legacy mode (deprecated).
    ncountours : int, optional, default=None
        Legacy parameter (typo) for number of contour levels.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    cbar : matplotlib.Colorbar object
        The corresponding colorbar object; None if no colorbar
        was requested. (suppressed in legacy mode)

    """
    if legacy:
        _warn(
            'Legacy mode is deprecated is will be removed in the'
            ' next major release. Until then use legacy=False',
            DeprecationWarning)
        cmap = _get_cmap(cmap)
        if offset != -1:
            _warn(
                'Parameter offset is deprecated and will be ignored',
                DeprecationWarning)
        if ncountours is not None:
            _warn(
                'Parameter ncountours is deprecated;'
                ' use ncontours instead',
                DeprecationWarning)
            ncontours = ncountours
        if vmin is None:
            vmin = 0.0
    else:
        if offset != -1:
            raise ValueError(
                'Parameter offset is not allowed outside legacy mode')
        if ncountours is not None:
            raise ValueError(
                'Parameter ncountours is not allowed outside'
                ' legacy mode; use ncontours instead')
    x, y, z = get_histogram(
        xall, yall, nbins=nbins, weights=weights,
        avoid_zero_count=avoid_zero_count)
    f = _to_free_energy(z, minener_zero=minener_zero) * kT
    fig, ax, cb = plot_map(
        x, y, f.T, ax=ax, cmap=cmap,
        ncontours=ncontours, vmin=vmin, vmax=vmax, levels=levels,
        cbar=cbar, cax=cax, cbar_label=cbar_label, logscale=logscale)
    if legacy:
        return fig, ax
    return fig, ax, cbar


def plot_contour(
        xall, yall, zall, ax=None, cmap='viridis',
        ncontours=100, vmin=None, vmax=None, levels=None,
        cbar=True, cax=None, cbar_label=None, logscale=False,
        nbins=100, method='nearest'):
    """Plot a two-dimensional contour map by interpolating
    scattered data on a grid.

    Parameters
    ----------
    xall : ndarray(T)
        Sample x-coordinates.
    yall : ndarray(T)
        Sample y-coordinates.
    zall : ndarray(T)
        Sample z-coordinates.
    ax : matplotlib.Axes object, optional, default=None
        The ax to plot to; if ax=None, a new ax (and fig) is created.
    cmap : matplotlib colormap, optional, default='Blues'
        The color map to use.
    ncontours : int, optional, default=100
        Number of contour levels.
    vmin : float, optional, default=None
        Lowest z-value to be plotted.
    vmax : float, optional, default=None
        Highest z-value to be plotted.
    levels : iterable of float, optional, default=None
        Contour levels to plot.
    cbar : boolean, optional, default=True
        Plot a color bar.
    cax : matplotlib.Axes object, optional, default=None
        Plot the colorbar into a custom axes object instead of
        stealing space from ax.
    cbar_label : str, optional, default=None
        Colorbar label string; use None to suppress it.
    logscale : boolean, optional, default=False
        Plot the z-values in logscale.
    nbins : int, optional, default=100
        Number of grid points used in each dimension.
    method : str, optional, default='nearest'
        Assignment method; scipy.interpolate.griddata supports the
        methods 'nearest', 'linear', and 'cubic'.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    cbar : matplotlib.Colorbar object
        The corresponding colorbar object; None if no colorbar
        was requested.

    """
    x, y, z = get_grid_data(
        xall, yall, zall, nbins=nbins, method=method)
    if vmin is None:
        vmin = _np.min(zall[zall > -_np.inf])
    if vmax is None:
        vmax = _np.max(zall[zall < _np.inf])
    if levels is None:
        eps = (vmax - vmin) / float(ncontours)
        levels = _np.linspace(vmin - eps, vmax + eps)
    return plot_map(
        x, y, z, ax=ax, cmap=cmap,
        ncontours=ncontours, vmin=None, vmax=None, levels=levels,
        cbar=cbar, cax=cax, cbar_label=cbar_label, logscale=logscale)
