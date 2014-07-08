import sys
import numpy as np

from scipy.interpolate import griddata

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

nx=80
ny=80

xedges=np.linspace(-180.0, 180.0, nx+1)
yedges=np.linspace(-180.0, 180.0, ny+1)

dx=xedges[1:]-xedges[0:-1]
dy=yedges[1:]-yedges[0:-1]

xcenters=xedges[0:-1]+0.5*dx
ycenters=yedges[0:-1]+0.5*dy

def stationary_distribution(centers, pi, levels=None, norm=LogNorm(),\
                                fmt='%.e', method='linear', fill_value=np.nan):
    r"""Make contourplot of alanine-dipeptide stationary distribution

    The scattered data is interpolated onto a regular grid before
    plotting.

    Parameters
    ----------
    centers : (N, 2) ndarray 
        (phi, psi) coordinates of MSM discretization.
    pi : (N, ) ndarray,
        Stationary vector

    
    """  
    X, Y=np.meshgrid(xcenters, ycenters)
    Z=np.abs(griddata(centers, pi, (X, Y), method=method, fill_value=fill_value))
    if levels is None:
        levels=10**(np.linspace(-5.0, -1.0, 20))
    V=np.asarray(levels)
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.linspace(-180.0, 180.0, 11))
    ax.set_yticks(np.linspace(-180.0, 180.0, 11))
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    cs=ax.contour(X, Y, Z, V, norm=norm)
    plt.clabel(cs, inline=1, fmt=fmt)
    plt.grid()

def free_energy(centers, A, levels=None, norm=None,\
                fmt='%.1f', method='linear', fill_value=np.nan,
                ax = None):
    r"""Make contourplot of alanine-dipeptide free energy.

    The scattered data is interpolated onto a regular grid 
    before plotting.

    Parameters
    ----------
    centers : (N, 2) ndarray 
        (phi, psi) coordinates of MSM discretization.
    A : (N, ) ndarray,
        Free energy.

    ax : optional matplotlib axis to plot to
    """
    X, Y=np.meshgrid(xcenters, ycenters)
    Z=griddata(centers, A, (X, Y), method=method, fill_value=fill_value)
    Z=Z-Z.min()
    if levels is None:
        levels=np.linspace(0.0, 50.0, 10)
    V=np.asarray(levels)
    if ax is None:
        fig=plt.figure()
        ax=fig.add_subplot(111)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.linspace(-180.0, 180.0, 11))
    ax.set_yticks(np.linspace(-180.0, 180.0, 11))
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    cs=ax.contour(X, Y, Z, V, norm=norm)
    plt.clabel(cs, inline=1, fmt=fmt)
    plt.grid()

# committor has same plot routine, but with other defaults
committor = lambda *args, **kwargs: \
   free_energy(*args, fill_value=0, method='cubic', **kwargs)


def eigenvector(centers, ev, levels=None, norm=None,\
                    fmt='%.e', method='linear', fill_value=np.nan):
    r"""Make contourplot of alanine-dipeptide stationary distribution

    The scattered data is interpolated onto a regular grid before
    plotting.

    Parameters
    ----------
    centers : (N, 2) ndarray 
        (phi, psi) coordinates of MSM discretization.
    ev : (N, ) ndarray,
        Right eigenvector
    
    """  
    X, Y=np.meshgrid(xcenters, ycenters)
    Z=griddata(centers, ev, (X, Y), method=method, fill_value=fill_value)
    if levels is None:
        levels=np.linspace(-0.1, 0.1, 21)
    V=np.asarray(levels)    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.linspace(-180.0, 180.0, 11))
    ax.set_yticks(np.linspace(-180.0, 180.0, 11))
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    cs=ax.contour(X, Y, Z, V, norm=norm)
    plt.clabel(cs, inline=1, fmt=fmt)
    plt.grid()

def eigenvalues(ev):
    r"""Plot eigenvalues of transition matrix
    
    Parameters
    ----------
    ev : (N, ) ndarray
        array of eigenvalues
    
    """
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel(r"$n$")
    ax.set_ylabel(r"$\lambda_n$")
    ax.plot(np.arange(len(ev)), ev, ls=' ', marker='o')

def pcca(centers, crisp):
    r"""Color sets according to their membership.

    Parameters
    ----------
    crisp : tuple of (M, ) ndarray
        crisp[0] contains the state index array and crisp[1]
        contains the membership number array.

    """
    states=crisp[0]
    member=crisp[1]

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlim(-180.0, 180.0)
    ax.set_ylim(-180.0, 180.0)
    ax.set_xticks(np.linspace(-180.0, 180.0, 11))
    ax.set_yticks(np.linspace(-180.0, 180.0, 11))
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    k=member.max()
    my_colors="bgrcyk"
    if k>len(my_colors):
        raise ValueError("There are too many different memberships to assign a unique color.")
    else:
        dx=18.0
        dy=18.0
        patches=[]
        for i in range (len(states)):
            x=centers[states[i], 0]
            y=centers[states[i], 1]
            col=my_colors[member[i]]            
            patch=plt.Rectangle((x, y), dx, dy, color=col)
            ax.add_patch(patch)                     

def timescale_histogram(ts, n=100):
    r"""Histogram of implied time scales.

    Parameters
    ----------
    ts : (M, ) ndarray
        Sample of implied time scales
    n : int (optional)
        Number of histogram bins

    """
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel(r"$t_i$")
    ax.set_ylabel(r"$p(t_i)$")
    ax.hist(ts, n, normed=True)


def amplitudes(a):
    r"""Plot fingerprint amplitudes.

    Parameters
    ----------
    a : (M,) ndarray
        Fingerprint amplitudes

    """
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.set_xlabel(r"$i$")
    ax.set_ylabel(r"$\gamma_i$")
    ax.plot(np.arange(len(a)), a)
