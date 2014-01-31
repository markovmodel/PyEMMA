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
                                fmt='%.1f', method='linear', fill_value=np.nan):
    r"""Make contourplot of alanine-dipeptide free energy.

    The scattered data is interpolated onto a regular grid 
    before plotting.

    Parameters
    ----------
    centers : (N, 2) ndarray 
        (phi, psi) coordinates of MSM discretization.
    A : (N, ) ndarray,
        Free energy.

    
    """  
    X, Y=np.meshgrid(xcenters, ycenters)
    Z=griddata(centers, A, (X, Y), method=method, fill_value=fill_value)
    Z=Z-Z.min()
    if levels is None:
         levels=np.linspace(0.0, 50.0, 10)
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
