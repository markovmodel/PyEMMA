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
import math
import numpy as np
from matplotlib import pylab as plt
from matplotlib import rcParams

__author__ = 'noe'


class NetworkPlot(object):

    def __init__(self, A, pos=None, xpos=None, ypos=None):
        r"""

        Parameters
        ----------
        A : ndarray(n,n)
            weight matrix or adjacency matrix of the network to visualize
        pos : ndarray(n,2)
            user-defined positions
        xpos : ndarray(n,)
            user-defined x-positions
        ypos : ndarray(n,)
            user-defined y-positions

        Examples
        --------
        We define first define a reactive flux by taking the following transition
        matrix and computing TPT from state 2 to 3.

        >>> import numpy as np
        >>> P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
        ...               [0.1,  0.75, 0.05, 0.05, 0.05],
        ...               [0.05,  0.1,  0.8,  0.0,  0.05],
        ...               [0.0,  0.2, 0.0,  0.8,  0.0],
        ...               [0.0,  0.02, 0.02, 0.0,  0.96]])
        >>> from pyemma import msm
        >>> F = msm.tpt(msm.markov_model(P), [2], [3])

        now plot the gross flux
        >>> NetworkPlot(F.gross_flux).plot_network() # doctest:+ELLIPSIS
        ...

        """
        if A.shape[0] >= 50:
            import warnings
            warnings.warn("The layout optimization method will take a long"
                          " time for large networks! It is recommended to"
                          " coarse grain your model first!")
        self.A = A
        self.pos = pos
        self.xpos = xpos
        self.ypos = ypos

    def _draw_arrow(self, x1, y1, x2, y2, Dx, Dy, label="", width=1.0,
                    arrow_curvature=1.0, color="grey",
                    patchA=None, patchB=None, shrinkA=0, shrinkB=0):
        """
        Draws a slightly curved arrow from (x1,y1) to (x2,y2).
        Will allow the given patches at start end end.

        """
        # set arrow properties
        dist = math.sqrt( ((x2-x1)/float(Dx))**2 + ((y2-y1)/float(Dy))**2 )
        arrow_curvature *= 0.075  # standard scale
        rad = arrow_curvature/(dist)
        tail_width = width
        head_width = max(0.5,2*width)
        head_length = head_width
        plt.annotate("",
                     xy=(x2, y2),
                     xycoords='data',
                     xytext=(x1, y1),
                     textcoords='data',
                     arrowprops=dict(arrowstyle='simple,head_length='+str(head_length)+',head_width='+str(head_width)+',tail_width='+str(tail_width),
                                     color=color, shrinkA=shrinkA, shrinkB=shrinkB, patchA=patchA, patchB=patchB,
                                     connectionstyle="arc3,rad=-"+str(rad)),
                     zorder=0)
        center = np.array([0.55*x1+0.45*x2, 0.55*y1+0.45*y2])  # weighted center position
        v = np.array([x2-x1,y2-y1])  # 1->2 vector
        vabs = np.abs(v)
        vnorm = np.array([v[1],-v[0]])  # orthogonal vector
        vnorm /= math.sqrt(np.dot(vnorm,vnorm))  # normalize
        z = np.cross(v,vnorm)  # cross product to determine the direction into which vnorm points
        if z < 0:
            vnorm *= -1
        offset = 0.5*arrow_curvature*( (vabs[0]/(vabs[0]+vabs[1]))*Dx + (vabs[1]/(vabs[0]+vabs[1]))*Dy )
        ptext = center + offset*vnorm
        plt.text(ptext[0], ptext[1], label, size=14, horizontalalignment='center', verticalalignment='center', zorder=1)

    def plot_network(self,
                     state_sizes = None, state_scale = 1.0, state_colors = '#ff5500',
                     arrow_scale = 1.0, arrow_curvature = 1.0, arrow_labels = 'weights', arrow_label_format='%10.2f',
                     max_width = 12, max_height = 12, figpadding = 0.2, xticks = False, yticks = False):
        """
        Draws a network using discs and curved arrows.

        The thicknesses and labels of the arrows are taken from the off-diagonal matrix elements in A.

        """
        if self.pos is None:
            self.layout_automatic()
        # number of nodes
        n = np.shape(self.A)[0]
        # get bounds and pad figure
        xmin = np.min(self.pos[:,0]); xmax = np.max(self.pos[:,0]); Dx = xmax-xmin
        xmin -= Dx*figpadding; xmax += Dx*figpadding; Dx *= 1+figpadding;
        ymin = np.min(self.pos[:,1]); ymax = np.max(self.pos[:,1]); Dy = ymax-ymin
        ymin -= Dy*figpadding; ymax += Dy*figpadding; Dy *= 1+figpadding;
        # sizes of nodes
        if state_sizes is None:
            state_sizes = 0.5 * state_scale * min(Dx,Dy)**2 * np.ones(n) / float(n)
        else:
            state_sizes = 0.5 * state_scale * min(Dx,Dy)**2 * state_sizes / (np.max(state_sizes) * float(n))
        # automatic arrow rescaling
        arrow_scale *= 1.0 / (np.max(self.A - np.diag(np.diag(self.A))) * math.sqrt(n))
        # size figure
        if (Dx/max_width > Dy/max_height):
            plt.figure(figsize=(max_width,Dy*(max_width/Dx)))
        else:
            plt.figure(figsize=(Dx*(max_height/Dy),max_height))
        # font sizes
        rcParams.update({'font.size': 20})
        # remove axis labels
        frame = plt.gca()
        if not xticks:
            frame.axes.get_xaxis().set_ticks([])
        if not yticks:
            frame.axes.get_yaxis().set_ticks([])
        # set node colors
        if state_colors is None:
            state_colors = '#ff5500'  # None is not acceptable
        if isinstance(state_colors, str):
            state_colors = [state_colors for i in xrange(n)]
        else:
            state_colors = [plt.cm.binary(int(256.0*state_colors[i])) for i in xrange(n)]  # transfrom from [0,1] to 255-scale
        # set arrow labels
        if isinstance(arrow_labels, np.ndarray):
            L = arrow_labels
        else:
            L = np.empty(np.shape(self.A), dtype=object)
        if arrow_labels is None:
            L[:,:] = ''
        elif arrow_labels.lower() == 'weights':
            for i in xrange(n):
                for j in xrange(n):
                    L[i,j] = arrow_label_format % self.A[i,j]
        else:
            raise ValueError('invalid arrow label format')

        # draw circles
        circles = []
        for i in xrange(n):
            fig = plt.gcf()
            # choose color
            c = plt.Circle(self.pos[i], radius=math.sqrt(0.5*state_sizes[i])/2.0 ,color=state_colors[i], zorder=2)
            circles.append(c)
            fig.gca().add_artist(c)
            # add annotation
            plt.text(self.pos[i][0], self.pos[i][1], str(i), size=14, horizontalalignment='center', verticalalignment='center',
                     color='black', zorder=3)

        # draw arrows
        for i in xrange(n):
            for j in xrange(i+1,n):
                if (abs(self.A[i,j]) > 0):
                    self._draw_arrow(self.pos[i,0], self.pos[i,1], self.pos[j,0], self.pos[j,1], Dx, Dy, label=str(L[i,j]),
                               width=arrow_scale*self.A[i,j], arrow_curvature=arrow_curvature,
                               patchA = circles[i], patchB = circles[j], shrinkA = 3, shrinkB = 0)
                if (abs(self.A[j,i]) > 0):
                    self._draw_arrow(self.pos[j,0], self.pos[j,1], self.pos[i,0], self.pos[i,1], Dx, Dy, label=str(L[j,i]),
                               width=arrow_scale*self.A[j,i], arrow_curvature=arrow_curvature,
                               patchA = circles[j], patchB = circles[i], shrinkA = 3, shrinkB = 0)

        # plot
        plt.xlim(xmin,xmax)
        plt.ylim(ymin,ymax)

    def _find_best_positions(self, V, g, nexplore, nstep_explore):
        """Finds best positions for the given grandalf graph nodes by minimizing a network potential

        """
        if (self.xpos is not None) and (self.ypos is not None):
            return np.array([self.xpos,self.ypos]), 0  # nothing to do
        from .grandalf.layouts import DigcoLayout
        class defaultview(object):
            w, h = 10, 10
            xy = None
        min_stress = float('infinity')
        best_pos = np.zeros((len(V),2))
        # explore
        for i in xrange(nexplore):
            for v in V:
                v.view = defaultview()
            dig = DigcoLayout(g.C[0], opt_x=(self.xpos is None), opt_y=(self.ypos is None))
            # pre-scale fixed variables in order to reduce unnecessary stress
            if self.xpos is not None:
                x = self.xpos * len(V) / (np.max(self.xpos) - np.min(self.xpos))
            else:
                x = None
            if self.ypos is not None:
                y = self.ypos * len(V) / (np.max(self.ypos) - np.min(self.ypos))
            else:
                y = None
            dig.init_all(x=x, y=y)
            dig.draw(N=nstep_explore)
            # print i, dig.stress
            if dig.stress < min_stress:
                for j in xrange(len(V)):
                    best_pos[j,:] = V[j].view.xy
                min_stress = dig.stress
        # rescale fixed to user settings and balance the other coordinate
        if self.xpos is not None:
            # rescale x to fixed value
            best_pos[:,0] *= (np.max(self.xpos) - np.min(self.xpos)) / (np.max(best_pos[:,0]) - np.min(best_pos[:,0]))
            best_pos[:,0] += np.min(self.xpos) - np.min(best_pos[:,0])
            # rescale y to balance
            if np.max(best_pos[:,1])-np.min(best_pos[:,1]) > 0.01:
                best_pos[:,1] *= (np.max(self.xpos) - np.min(self.xpos)) / (np.max(best_pos[:,1]) - np.min(best_pos[:,1]))
        if self.ypos is not None:
            best_pos[:,1] *= (np.max(self.ypos) - np.min(self.ypos)) / (np.max(best_pos[:,1]) - np.min(best_pos[:,1]))
            best_pos[:,1] += np.min(self.ypos) - np.min(best_pos[:,1])
            # rescale x to balance
            if np.max(best_pos[:,0])-np.min(best_pos[:,0]) > 0.01:
                best_pos[:,0] *= (np.max(self.ypos) - np.min(self.ypos)) / (np.max(best_pos[:,0]) - np.min(best_pos[:,0]))
        # re-order
        neworder = np.zeros(len(V),dtype=int)
        for i,v in enumerate(g.C[0].sV):
            neworder[i] = v.data
        best_pos = best_pos[neworder,:]
        # done
        return best_pos, min_stress

    def layout_automatic(self):
        n = np.shape(self.A)[0]
        I,J = np.where(self.A>0)
        # create graph object
        from .grandalf.graphs import Vertex,Edge,Graph
        V = [Vertex(i) for i in xrange(n)]
        E = [Edge(V[I[i]],V[J[i]]) for i in xrange(len(I))]
        g = Graph(V,E)
        # layout
        self.pos, self.stress = self._find_best_positions(V, g, 10, 25)
        # print 'best stress = ',stress


def plot_markov_model(P, pos = None, state_sizes = None, state_scale = 1.0, state_colors = '#ff5500', minflux = 1e-6,
                      arrow_scale = 1.0, arrow_curvature = 1.0, arrow_labels = 'weights', arrow_label_format='%10.2f',
                      max_width = 12, max_height = 12, figpadding = 0.2):
    r"""Plots a network representation of a Markov model transition matrix

    This visualization is not optimized for large matrices. It is meant to be used for the visualization of small
    models with up to 10-20 states, e.g. obtained by a HMM coarse-graining. If used with large network, the automatic
    node positioning will be very slow and may still look ugly.

    Parameters
    ----------
    P : ndarray(n,n) or MSM object with attribute 'transition matrix'
        Transition matrix or MSM object
    pos : ndarray(n,2), optional, default=None
        User-defined positions to draw the states on. If not given, will try to place them automatically.
    state_sizes : ndarray(n), optional, default=None
        User-defined areas of the discs drawn for each state. If not given, the stationary probability of P
        will be used
    state_colors : string or ndarray(n), optional, default='#ff5500' (orange)
        Either a string with a Hex code for a single color used for all states, or an array of values in [0,1]
        which will result in a grayscale plot
    minflux : float, optional, default=1e-6
        The minimal flux (p_i * p_ij) for a transition to be drawn
    arrow_scale : float, optional, default=1.0
        Relative arrow scale. Set to a value different from 1 to increase or decrease the arrow width.
    arrow_curvature : float, optional, default=1.0
        Relative arrow curvature. Set to a value different from 1 to make arrows more or less curved.
    arrow_labels : 'weights', None or a ndarray(n,n) with label strings. Optional, default='weights'
        Strings to be placed upon arrows. If None, no labels will be used. If 'weights', the elements of P will
        be used. If a matrix of strings is given by the user these will be used.
    arrow_label_format : str, optional, default='%10.2f'
        The numeric format to print the arrow labels
    max_width = 12
        The maximum figure width
    max_height = 12
        The maximum figure height
    figpadding = 0.2
        The relative figure size used for the padding

    Returns
    -------
    The positions of states. Can be used later to plot a different network representation (e.g. the flux)

    Examples
    --------
    >>> P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
    ...              [0.1,  0.75, 0.05, 0.05, 0.05],
    ...              [0.05,  0.1,  0.8,  0.0,  0.05],
    ...              [0.0,  0.2, 0.0,  0.8,  0.0],
    ...              [0.0,  0.02, 0.02, 0.0,  0.96]])
    >>> plot_markov_model(P) # doctest:+ELLIPSIS
    array...

    """
    from pyemma.msm import analysis as msmana
    if isinstance(P, np.ndarray):
        P = P.copy()
    else:
        # MSM object? then get transition matrix first
        P = P.transition_matrix.copy()
    if state_sizes is None:
        state_sizes = msmana.stationary_distribution(P)
    if minflux > 0:
        F = np.dot(np.diag(msmana.stationary_distribution(P)), P)
        I,J = np.where(F < minflux)
        P[I,J] = 0.0
    plot = NetworkPlot(P, pos=pos)
    if pos is not None:
        plot.layout_automatic()
    plot.plot_network(state_sizes=state_sizes, state_scale=state_scale, state_colors=state_colors,
                      arrow_scale=arrow_scale, arrow_curvature=arrow_curvature, arrow_labels=arrow_labels,
                      arrow_label_format=arrow_label_format, max_width=max_width, max_height=max_height,
                      figpadding=figpadding, xticks=False, yticks=False)
    plt.show()
    return plot.pos

def plot_flux(flux, pos = None, state_sizes = None, state_scale = 1.0, state_colors = '#ff5500', minflux = 1e-9,
              arrow_scale = 1.0, arrow_curvature = 1.0, arrow_labels = 'weights', arrow_label_format='%10.2f',
              max_width = 12, max_height = 12, figpadding = 0.2, attribute_to_plot='net_flux'):
    r"""Plots a network representation of the reactive flux

    This visualization is not optimized for large fluxes. It is meant to be used for the visualization of small
    models with up to 10-20 states, e.g. obtained by a PCCA-based coarse-graining of the full flux.
    If used with large network, the automatic node positioning will be very slow and may still look ugly.

    Parameters
    ----------
    flux : :class:`ReactiveFlux <pyemma.msm.flux.ReactiveFlux>`
        reactive flux object
    pos : ndarray(n,2), optional, default=None
        User-defined positions to draw the states on. If not given, will set the x coordinates equal to the
        committor probability and try to place the y coordinates automatically
    state_sizes : ndarray(n), optional, default=None
        User-defined areas of the discs drawn for each state. If not given, the stationary probability of P
        will be used
    state_colors : string or ndarray(n), optional, default='#ff5500' (orange)
        Either a string with a Hex code for a single color used for all states, or an array of values in [0,1]
        which will result in a grayscale plot
    minflux : float, optional, default=1e-9
        The minimal flux for a transition to be drawn
    arrow_scale : float, optional, default=1.0
        Relative arrow scale. Set to a value different from 1 to increase or decrease the arrow width.
    arrow_curvature : float, optional, default=1.0
        Relative arrow curvature. Set to a value different from 1 to make arrows more or less curved.
    arrow_labels : 'weights', None or a ndarray(n,n) with label strings. Optional, default='weights'
        Strings to be placed upon arrows. If None, no labels will be used. If 'weights', the elements of P will
        be used. If a matrix of strings is given by the user these will be used.
    arrow_label_format : str, optional, default='%10.2f'
        The numeric format to print the arrow labels
    max_width : int (default = 12)
        The maximum figure width
    max_height: int (default = 12)
        The maximum figure height
    figpadding: float (default = 0.2)
        The relative figure size used for the padding

    Returns
    -------
    The positions of states. Can be used later to plot a different network representation (e.g. the flux)

    Examples
    --------
    We define first define a reactive flux by taking the following transition matrix and computing TPT from state 2 to 3

    >>> import numpy as np
    >>> P = np.array([[0.8,  0.15, 0.05,  0.0,  0.0],
    ...               [0.1,  0.75, 0.05, 0.05, 0.05],
    ...               [0.05,  0.1,  0.8,  0.0,  0.05],
    ...               [0.0,  0.2, 0.0,  0.8,  0.0],
    ...               [0.0,  0.02, 0.02, 0.0,  0.96]])
    >>> from pyemma import msm
    >>> F = msm.tpt(msm.markov_model(P), [2], [3])
    >>> F.flux[:] *= 100

    Scale the flux by 100 is basically a change of units to get numbers close to 1 (avoid printing many zeros).
    Now we visualize the flux:

    >>> plot_flux(F) # doctest:+ELLIPSIS
    array...

    """
    F = getattr(flux, attribute_to_plot)
    if minflux > 0:
        I, J = np.where(F < minflux)
        F[I, J] = 0.0
    c = flux.committor
    if state_sizes is None:
        state_sizes = flux.stationary_distribution
    plot = NetworkPlot(F, pos=pos, xpos=c)
    if pos is not None:
        plot.layout_automatic()
    plot.plot_network(state_sizes=state_sizes, state_scale=state_scale, state_colors=state_colors,
                      arrow_scale=arrow_scale, arrow_curvature=arrow_curvature, arrow_labels=arrow_labels,
                      arrow_label_format=arrow_label_format, max_width=max_width, max_height=max_height,
                      figpadding=figpadding, xticks=True, yticks=False)
    plt.xlabel('Committor probability')
    plt.show()
    return plot.pos
