# encoding: utf-8

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


import numpy as _np
from warnings import warn as _warn

__author__ = 'noe'

_matplotlib_version = None


def _get_cmap(cmap):
    # matplotlib 2.0 deprecated 'spectral' colormap, renamed to nipy_spectral.
    global _matplotlib_version
    if _matplotlib_version is None:
        from matplotlib import __version__
        from distutils.version import LooseVersion
        _matplotlib_version = LooseVersion(__version__).version
    if cmap == 'spectral' and _matplotlib_version >= (2, ):
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
        ncontours=ncontours, vmin=vmin, vmax=vmax, levels='legacy',
        cbar=colorbar, cax=None, cbar_label=None, norm=None,
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
    ncontours : int, optional, default=50
        number of contour levels
    fig : matplotlib Figure object, optional, default=None
        the figure to plot into. When set to None the default
        Figure object will be used
    ax : matplotlib Axes object, optional, default=None
        the axes to plot to. When set to None the default Axes
        object will be used.
    cmap : matplotlib colormap, optional, default=None
        the color map to use. None will use pylab.cm.jet.
    outfile : str, optional, default=None
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
    return x, y, z.T # transpose to match x/y-directions


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


def _prune_kwargs(kwargs):
    """Remove non-allowed keys from a kwargs dictionary.

    Parameters
    ----------
    kwargs : dict
        Named parameters to prune.

    """
    allowed_keys = [
        'corner_mask', 'alpha', 'locator', 'extend', 'xunits',
        'yunits', 'antialiased', 'nchunk', 'hatches', 'zorder']
    ignored = [key for key in kwargs.keys() if key not in allowed_keys]
    for key in ignored:
        _warn(
            '{}={} is not an allowed optional parameter and will'
            ' be ignored'.format(key, kwargs[key]))
        kwargs.pop(key, None)
    return kwargs


def plot_map(
        x, y, z, ax=None, cmap=None,
        ncontours=100, vmin=None, vmax=None, levels=None,
        cbar=True, cax=None, cbar_label=None,
        cbar_orientation='vertical', norm=None,
        **kwargs):
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
    cmap : matplotlib colormap, optional, default=None
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
    cbar_orientation : str, optional, default='vertical'
        Colorbar orientation; choose 'vertical' or 'horizontal'.
    norm : matplotlib norm, optional, default=None
        Use a norm when coloring the contour plot.

    Optional parameters for contourf (**kwargs)
    -------------------------------------------
    corner_mask : boolean, optional
        Enable/disable corner masking, which only has an effect if
        z is a masked array. If False, any quad touching a masked
        point is masked out. If True, only the triangular corners
        of quads nearest those points are always masked out, other
        triangular corners comprising three unmasked points are
        contoured as usual.
        Defaults to rcParams['contour.corner_mask'], which
        defaults to True.
    alpha : float
        The alpha blending value.
    locator : [ None | ticker.Locator subclass ]
        If locator is None, the default MaxNLocator is used. The
        locator is used to determine the contour levels if they are
        not given explicitly via the levels argument.
    extend : [ ‘neither’ | ‘both’ | ‘min’ | ‘max’ ]
        Unless this is ‘neither’, contour levels are automatically
        added to one or both ends of the range so that all data are
        included. These added ranges are then mapped to the special
        colormap values which default to the ends of the
        colormap range, but can be set via
        matplotlib.colors.Colormap.set_under() and
        matplotlib.colors.Colormap.set_over() methods.
    xunits, yunits : [ None | registered units ]
        Override axis units by specifying an instance of a
        matplotlib.units.ConversionInterface.
    antialiased : boolean, optional
        Enable antialiasing, overriding the defaults. For filled
        contours, the default is True. For line contours, it is
        taken from rcParams[‘lines.antialiased’].
    nchunk : [ 0 | integer ]
        If 0, no subdivision of the domain. Specify a positive
        integer to divide the domain into subdomains of nchunk by
        nchunk quads. Chunking reduces the maximum length of polygons
        generated by the contouring algorithm which reduces the
        rendering workload passed on to the backend and also requires
        slightly less RAM. It can however introduce rendering
        artifacts at chunk boundaries depending on the backend, the
        antialiased flag and value of alpha.
    hatches :
        A list of cross hatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour. Hatching
        is supported in the PostScript, PDF, SVG and Agg backends
        only.
    zorder : float
        Set the zorder for the artist. Artists with lower zorder
        values are drawn first.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    misc : dict
        Contains a matplotlib.contour.QuadContourSet 'mappable' and,
        if requested, a matplotlib.Colorbar object 'cbar'.

    """
    import matplotlib.pyplot as _plt
    if ax is None:
        fig, ax = _plt.subplots()
    else:
        fig = ax.get_figure()
    mappable = ax.contourf(
        x, y, z, ncontours, norm=norm,
        vmin=vmin, vmax=vmax, cmap=cmap,
        levels=levels, **_prune_kwargs(kwargs))
    misc = dict(mappable=mappable)
    if cbar_orientation not in ('horizontal', 'vertical'):
        raise ValueError(
            'cbar_orientation must be "horizontal" or "vertical"')
    if cbar:
        if cax is None:
            cbar_ = fig.colorbar(
                mappable, ax=ax, orientation=cbar_orientation)
        else:
            cbar_ = fig.colorbar(
                mappable, cax=cax, orientation=cbar_orientation)
        if cbar_label is not None:
            cbar_.set_label(cbar_label)
        misc.update(cbar=cbar_)
    return fig, ax, misc


# ######################################################################
#
# new plotting functions
#
# ######################################################################


def plot_density(
        xall, yall, ax=None, cmap=None,
        ncontours=100, vmin=None, vmax=None, levels=None,
        cbar=True, cax=None, cbar_label='sample density',
        cbar_orientation='vertical', logscale=False, nbins=100,
        weights=None, avoid_zero_count=False, **kwargs):
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
    cmap : matplotlib colormap, optional, default=None
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
    cbar_label : str, optional, default='sample density'
        Colorbar label string; use None to suppress it.
    cbar_orientation : str, optional, default='vertical'
        Colorbar orientation; choose 'vertical' or 'horizontal'.
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

    Optional parameters for contourf (**kwargs)
    -------------------------------------------
    corner_mask : boolean, optional
        Enable/disable corner masking, which only has an effect if
        z is a masked array. If False, any quad touching a masked
        point is masked out. If True, only the triangular corners
        of quads nearest those points are always masked out, other
        triangular corners comprising three unmasked points are
        contoured as usual.
        Defaults to rcParams['contour.corner_mask'], which
        defaults to True.
    alpha : float
        The alpha blending value.
    locator : [ None | ticker.Locator subclass ]
        If locator is None, the default MaxNLocator is used. The
        locator is used to determine the contour levels if they are
        not given explicitly via the levels argument.
    extend : [ ‘neither’ | ‘both’ | ‘min’ | ‘max’ ]
        Unless this is ‘neither’, contour levels are automatically
        added to one or both ends of the range so that all data are
        included. These added ranges are then mapped to the special
        colormap values which default to the ends of the
        colormap range, but can be set via
        matplotlib.colors.Colormap.set_under() and
        matplotlib.colors.Colormap.set_over() methods.
    xunits, yunits : [ None | registered units ]
        Override axis units by specifying an instance of a
        matplotlib.units.ConversionInterface.
    antialiased : boolean, optional
        Enable antialiasing, overriding the defaults. For filled
        contours, the default is True. For line contours, it is
        taken from rcParams[‘lines.antialiased’].
    nchunk : [ 0 | integer ]
        If 0, no subdivision of the domain. Specify a positive
        integer to divide the domain into subdomains of nchunk by
        nchunk quads. Chunking reduces the maximum length of polygons
        generated by the contouring algorithm which reduces the
        rendering workload passed on to the backend and also requires
        slightly less RAM. It can however introduce rendering
        artifacts at chunk boundaries depending on the backend, the
        antialiased flag and value of alpha.
    hatches :
        A list of cross hatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour. Hatching
        is supported in the PostScript, PDF, SVG and Agg backends
        only.
    zorder : float
        Set the zorder for the artist. Artists with lower zorder
        values are drawn first.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    misc : dict
        Contains a matplotlib.contour.QuadContourSet 'mappable' and,
        if requested, a matplotlib.Colorbar object 'cbar'.

    """
    x, y, z = get_histogram(
        xall, yall, nbins=nbins, weights=weights,
        avoid_zero_count=avoid_zero_count)
    pi = _to_density(z)
    pi = _np.ma.masked_where(pi <= 0, pi)
    if logscale:
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin, vmax=vmax)
        if levels is None:
            levels = _np.logspace(
                _np.floor(_np.log10(pi.min())),
                _np.ceil(_np.log10(pi.max())),
                ncontours + 1)
    else:
        norm = None
    fig, ax, misc = plot_map(
        x, y, pi, ax=ax, cmap=cmap,
        ncontours=ncontours, vmin=vmin, vmax=vmax, levels=levels,
        cbar=cbar, cax=cax, cbar_label=cbar_label,
        cbar_orientation=cbar_orientation, norm=norm,
        **kwargs)
    if cbar and logscale:
        from matplotlib.ticker import LogLocator
        misc['cbar'].set_ticks(LogLocator(base=10.0, subs=range(10)))
    return fig, ax, misc


def plot_free_energy(
        xall, yall, weights=None, ax=None, nbins=100, ncontours=100,
        offset=-1, avoid_zero_count=False, minener_zero=True, kT=1.0,
        vmin=None, vmax=None, cmap='nipy_spectral', cbar=True,
        cbar_label='free energy / kT', cax=None, levels=None,
        legacy=True, ncountours=None, cbar_orientation='vertical',
        **kwargs):
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
    legacy : boolean, optional, default=True
        Switch to use the function in legacy mode (deprecated).
    ncountours : int, optional, default=None
        Legacy parameter (typo) for number of contour levels.
    cbar_orientation : str, optional, default='vertical'
        Colorbar orientation; choose 'vertical' or 'horizontal'.

    Optional parameters for contourf (**kwargs)
    -------------------------------------------
    corner_mask : boolean, optional
        Enable/disable corner masking, which only has an effect if
        z is a masked array. If False, any quad touching a masked
        point is masked out. If True, only the triangular corners
        of quads nearest those points are always masked out, other
        triangular corners comprising three unmasked points are
        contoured as usual.
        Defaults to rcParams['contour.corner_mask'], which
        defaults to True.
    alpha : float
        The alpha blending value.
    locator : [ None | ticker.Locator subclass ]
        If locator is None, the default MaxNLocator is used. The
        locator is used to determine the contour levels if they are
        not given explicitly via the levels argument.
    extend : [ ‘neither’ | ‘both’ | ‘min’ | ‘max’ ]
        Unless this is ‘neither’, contour levels are automatically
        added to one or both ends of the range so that all data are
        included. These added ranges are then mapped to the special
        colormap values which default to the ends of the
        colormap range, but can be set via
        matplotlib.colors.Colormap.set_under() and
        matplotlib.colors.Colormap.set_over() methods.
    xunits, yunits : [ None | registered units ]
        Override axis units by specifying an instance of a
        matplotlib.units.ConversionInterface.
    antialiased : boolean, optional
        Enable antialiasing, overriding the defaults. For filled
        contours, the default is True. For line contours, it is
        taken from rcParams[‘lines.antialiased’].
    nchunk : [ 0 | integer ]
        If 0, no subdivision of the domain. Specify a positive
        integer to divide the domain into subdomains of nchunk by
        nchunk quads. Chunking reduces the maximum length of polygons
        generated by the contouring algorithm which reduces the
        rendering workload passed on to the backend and also requires
        slightly less RAM. It can however introduce rendering
        artifacts at chunk boundaries depending on the backend, the
        antialiased flag and value of alpha.
    hatches :
        A list of cross hatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour. Hatching
        is supported in the PostScript, PDF, SVG and Agg backends
        only.
    zorder : float
        Set the zorder for the artist. Artists with lower zorder
        values are drawn first.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    misc : dict
        Contains a matplotlib.contour.QuadContourSet 'mappable' and,
        if requested, a matplotlib.Colorbar object 'cbar'.

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
    fig, ax, misc = plot_map(
        x, y, f, ax=ax, cmap=cmap,
        ncontours=ncontours, vmin=vmin, vmax=vmax, levels=levels,
        cbar=cbar, cax=cax, cbar_label=cbar_label,
        cbar_orientation=cbar_orientation, norm=None,
        **kwargs)
    if legacy:
        return fig, ax
    return fig, ax, misc


def plot_contour(
        xall, yall, zall, ax=None, cmap=None,
        ncontours=100, vmin=None, vmax=None, levels=None,
        cbar=True, cax=None, cbar_label=None,
        cbar_orientation='vertical', norm=None, nbins=100,
        method='nearest', mask=False, **kwargs):
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
    cmap : matplotlib colormap, optional, default=None
        The color map to use.
    ncontours : int, optional, default=100
        Number of contour levels.
    vmin : float, optional, default=None
        Lowest z-value to be plotted.
    vmax : float, optional, default=None
        Highest z-value to be plotted.
    levels : iterable of float, optional, default=None
        Contour levels to plot; use legacy style calculation
        if 'legacy'.
    cbar : boolean, optional, default=True
        Plot a color bar.
    cax : matplotlib.Axes object, optional, default=None
        Plot the colorbar into a custom axes object instead of
        stealing space from ax.
    cbar_label : str, optional, default=None
        Colorbar label string; use None to suppress it.
    cbar_orientation : str, optional, default='vertical'
        Colorbar orientation; choose 'vertical' or 'horizontal'.
    norm : matplotlib norm, optional, default=None
        Use a norm when coloring the contour plot.
    nbins : int, optional, default=100
        Number of grid points used in each dimension.
    method : str, optional, default='nearest'
        Assignment method; scipy.interpolate.griddata supports the
        methods 'nearest', 'linear', and 'cubic'.
    mask : boolean, optional, default=False
        Hide unsampled areas if True.

    Optional parameters for contourf (**kwargs)
    -------------------------------------------
    corner_mask : boolean, optional
        Enable/disable corner masking, which only has an effect if
        z is a masked array. If False, any quad touching a masked
        point is masked out. If True, only the triangular corners
        of quads nearest those points are always masked out, other
        triangular corners comprising three unmasked points are
        contoured as usual.
        Defaults to rcParams['contour.corner_mask'], which
        defaults to True.
    alpha : float
        The alpha blending value.
    locator : [ None | ticker.Locator subclass ]
        If locator is None, the default MaxNLocator is used. The
        locator is used to determine the contour levels if they are
        not given explicitly via the levels argument.
    extend : [ ‘neither’ | ‘both’ | ‘min’ | ‘max’ ]
        Unless this is ‘neither’, contour levels are automatically
        added to one or both ends of the range so that all data are
        included. These added ranges are then mapped to the special
        colormap values which default to the ends of the
        colormap range, but can be set via
        matplotlib.colors.Colormap.set_under() and
        matplotlib.colors.Colormap.set_over() methods.
    xunits, yunits : [ None | registered units ]
        Override axis units by specifying an instance of a
        matplotlib.units.ConversionInterface.
    antialiased : boolean, optional
        Enable antialiasing, overriding the defaults. For filled
        contours, the default is True. For line contours, it is
        taken from rcParams[‘lines.antialiased’].
    nchunk : [ 0 | integer ]
        If 0, no subdivision of the domain. Specify a positive
        integer to divide the domain into subdomains of nchunk by
        nchunk quads. Chunking reduces the maximum length of polygons
        generated by the contouring algorithm which reduces the
        rendering workload passed on to the backend and also requires
        slightly less RAM. It can however introduce rendering
        artifacts at chunk boundaries depending on the backend, the
        antialiased flag and value of alpha.
    hatches :
        A list of cross hatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour. Hatching
        is supported in the PostScript, PDF, SVG and Agg backends
        only.
    zorder : float
        Set the zorder for the artist. Artists with lower zorder
        values are drawn first.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    misc : dict
        Contains a matplotlib.contour.QuadContourSet 'mappable' and,
        if requested, a matplotlib.Colorbar object 'cbar'.

    """
    x, y, z = get_grid_data(
        xall, yall, zall, nbins=nbins, method=method)
    if vmin is None:
        vmin = _np.min(zall[zall > -_np.inf])
    if vmax is None:
        vmax = _np.max(zall[zall < _np.inf])
    if levels == 'legacy':
        eps = (vmax - vmin) / float(ncontours)
        levels = _np.linspace(vmin - eps, vmax + eps)
    if mask:
        _, _, counts = get_histogram(
            xall, yall, nbins=nbins, weights=None,
            avoid_zero_count=None)
        z = _np.ma.masked_where(counts.T <= 0, z)
    return plot_map(
        x, y, z, ax=ax, cmap=cmap,
        ncontours=ncontours, vmin=vmin, vmax=vmax, levels=levels,
        cbar=cbar, cax=cax, cbar_label=cbar_label,
        cbar_orientation=cbar_orientation, norm=norm,
        **kwargs)


def plot_state_map(
        xall, yall, states, ax=None, ncontours=100, cmap=None,
        cbar=True, cax=None, cbar_label='state',
        cbar_orientation='vertical', nbins=100, mask=True,
        **kwargs):
    """Plot a two-dimensional contour map of states by interpolating
    labels of scattered data on a grid.

    Parameters
    ----------
    xall : ndarray(T)
        Sample x-coordinates.
    yall : ndarray(T)
        Sample y-coordinates.
    states : ndarray(T)
        Sample state labels.
    ax : matplotlib.Axes object, optional, default=None
        The ax to plot to; if ax=None, a new ax (and fig) is created.
    cmap : matplotlib colormap, optional, default=None
        The color map to use.
    ncontours : int, optional, default=100
        Number of contour levels.
    cbar : boolean, optional, default=True
        Plot a color bar.
    cax : matplotlib.Axes object, optional, default=None
        Plot the colorbar into a custom axes object instead of
        stealing space from ax.
    cbar_label : str, optional, default='state'
        Colorbar label string; use None to suppress it.
    cbar_orientation : str, optional, default='vertical'
        Colorbar orientation; choose 'vertical' or 'horizontal'.
    nbins : int, optional, default=100
        Number of grid points used in each dimension.
    mask : boolean, optional, default=False
        Hide unsampled areas is True.

    Optional parameters for contourf (**kwargs)
    -------------------------------------------
    corner_mask : boolean, optional
        Enable/disable corner masking, which only has an effect if
        z is a masked array. If False, any quad touching a masked
        point is masked out. If True, only the triangular corners
        of quads nearest those points are always masked out, other
        triangular corners comprising three unmasked points are
        contoured as usual.
        Defaults to rcParams['contour.corner_mask'], which
        defaults to True.
    alpha : float
        The alpha blending value.
    locator : [ None | ticker.Locator subclass ]
        If locator is None, the default MaxNLocator is used. The
        locator is used to determine the contour levels if they are
        not given explicitly via the levels argument.
    extend : [ ‘neither’ | ‘both’ | ‘min’ | ‘max’ ]
        Unless this is ‘neither’, contour levels are automatically
        added to one or both ends of the range so that all data are
        included. These added ranges are then mapped to the special
        colormap values which default to the ends of the
        colormap range, but can be set via
        matplotlib.colors.Colormap.set_under() and
        matplotlib.colors.Colormap.set_over() methods.
    xunits, yunits : [ None | registered units ]
        Override axis units by specifying an instance of a
        matplotlib.units.ConversionInterface.
    antialiased : boolean, optional
        Enable antialiasing, overriding the defaults. For filled
        contours, the default is True. For line contours, it is
        taken from rcParams[‘lines.antialiased’].
    nchunk : [ 0 | integer ]
        If 0, no subdivision of the domain. Specify a positive
        integer to divide the domain into subdomains of nchunk by
        nchunk quads. Chunking reduces the maximum length of polygons
        generated by the contouring algorithm which reduces the
        rendering workload passed on to the backend and also requires
        slightly less RAM. It can however introduce rendering
        artifacts at chunk boundaries depending on the backend, the
        antialiased flag and value of alpha.
    hatches :
        A list of cross hatch patterns to use on the filled areas.
        If None, no hatching will be added to the contour. Hatching
        is supported in the PostScript, PDF, SVG and Agg backends
        only.
    zorder : float
        Set the zorder for the artist. Artists with lower zorder
        values are drawn first.

    Returns
    -------
    fig : matplotlib.Figure object
        The figure in which the used ax resides.
    ax : matplotlib.Axes object
        The ax in which the map was plotted.
    misc : dict
        Contains a matplotlib.contour.QuadContourSet 'mappable' and,
        if requested, a matplotlib.Colorbar object 'cbar'.

    Notes
    -----
    Please note that this plot is an approximative visualization:
    the underlying matplotlib.countourf function smoothes transitions
    between different values and, thus, coloring at state boundaries
    might be imprecise.

    """
    from matplotlib.cm import get_cmap
    nstates = int(_np.max(states) + 1)
    cmap_ = get_cmap(cmap, nstates)
    fig, ax, misc = plot_contour(
        xall, yall, states, ax=ax, cmap=cmap_,
        ncontours=ncontours, vmin=None, vmax=None, levels=None,
        cbar=cbar, cax=cax, cbar_label=cbar_label,
        cbar_orientation=cbar_orientation, norm=None, nbins=nbins,
        method='nearest', mask=mask, **kwargs)
    if cbar:
        cmin, cmax = misc['mappable'].get_clim()
        f = (cmax - cmin) / float(nstates)
        n = _np.arange(nstates)
        misc['cbar'].set_ticks((n + 0.5) * f)
        misc['cbar'].set_ticklabels(n)
    return fig, ax, misc
