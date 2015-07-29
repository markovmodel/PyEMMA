
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__author__ = 'noe'

import numpy as _np
import matplotlib.pylab as _plt
from scipy.interpolate import griddata as gd

def contour(x, y, z, ncontours = 50, colorbar=True, fig=None, ax=None, method='linear', zlim=None, cmap=None):
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
    ax = contour(x, y, z, ncontours=ncontours, colorbar=colorbar, fig=fig, ax=ax, cmap=cmap)

    # scatter points
    ax.scatter(x,y,marker='o',c='b',s=5)

    # show or save
    if outfile is not None:
        _plt.savefig(outfile)

    return ax

