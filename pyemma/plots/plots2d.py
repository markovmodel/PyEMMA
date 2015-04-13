__author__ = 'noe'

import numpy as _np
import matplotlib.pylab as _plt
from scipy.interpolate import griddata as gd

def scatter_contour(x, y, z, ncontours = 50, colorbar=True, fig=None, ax=None, cmap=None, outfile=None):
    """Shows a contour plot on scattered data (x,y,z) and the plots the positions of the points (x,y) on top.

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
    # check input
    if (ax is None):
        if fig is None:
            ax = _plt.gca()
        else:
            ax = fig.gca()

    # grid data
    points = _np.hstack([x[:,None],y[:,None]])
    xi, yi = _np.mgrid[x.min():x.max():100j, y.min():y.max():200j]
    zi = gd(points, z, (xi, yi), method='cubic')
    # contour level levels
    eps = (z.max() - z.min()) / float(ncontours)
    levels = _np.linspace(z.min() - eps, z.max() + eps)
    # contour plot
    if cmap is None:
        cmap=_plt.cm.jet
    cf = ax.contourf(xi, yi, zi, 15, cmap=cmap, levels=levels)
    # color bar if requested
    if colorbar:
        _plt.colorbar(cf, ax=ax)
    # scatter points
    ax.scatter(x,y,marker='o',c='b',s=5)

    # show or save
    #if outfile is None:
    #    _plt.show()
    if outfile is not None:
        _plt.savefig(outfile)

    return ax