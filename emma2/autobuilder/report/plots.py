'''
Created on Dec 8, 2013

@author: noe
'''

import matplotlib.pyplot as plt
import numpy as np
import math
import sys


colors = ['black','red','green','blue','orange','cyan','brown','grey','magenta','yellow']



def plot_list(Ylist, log_x = False, log_y = False, xlabel = None, ylabel = None, title = None, outfile = None):
    """
    Returns a subplot, or writes to a file
    """
    ny = len(Ylist)
    
    f, ax = plt.subplots()
    if (title != None):
        ax.set_title(title)
    # Log scale if requested
    if (log_x):
        plt.xscale('log')
    if (log_y):
        plt.yscale('log')
    # Plot all lines
    for i in range(0,ny):
        y = Ylist[i]
        x = range(len(y))
        icolor = i % len(colors)
        ax.plot(x, y, color=colors[icolor], lw=2)
    # Axis labels
    if (xlabel != None):
        ax.set_xlabel(xlabel)
    if (ylabel != None):
        ax.set_ylabel(ylabel)
    # output
    if (outfile != None):
        plt.savefig(outfile)
    return ax


def plot_mult(X, Y, log_x = False, log_y = False, xlabel = None, ylabel = None, outfile = None):
    """
    Returns a subplot, or writes to a file
    """
    ny = len(Y[0])
    
    f, ax = plt.subplots()
    ax.set_title('Implied timescales')
    # Log scale if requested
    if (log_x):
        plt.xscale('log')
    if (log_y):
        plt.yscale('log')
    # Plot all lines
    for i in range(0,ny):
        y = Y[:,i]
        icolor = i % len(colors)
        ax.plot(X, y, color=colors[icolor], lw=2)
    # Axis labels
    if (xlabel != None):
        ax.set_xlabel(xlabel)
    if (ylabel != None):
        ax.set_ylabel(ylabel)
    # output
    if (outfile != None):
        plt.savefig(outfile)
    return ax


###############################################################################
# IMPLIED TIMESCALES
###############################################################################

def plot_implied_timescales(lags, timescales, log_x = False, log_y = True, xlabel = 'lag time (steps)', ylabel = 'timescale (steps)', plot_forbidden = True, outfile = None):
    """
    Returns a subplot, or writes to a file
    """
    ntimescales = len(timescales[0])
    
    f, ax = plt.subplots()
    ax.set_title('Implied timescales')
    # Log scale if requested
    if (log_x):
        plt.xscale('log')
    if (log_y):
        plt.yscale('log')
    # Plot all lines
    for i in range(0,ntimescales):
        y = timescales[:,i]
        icolor = i % len(colors)
        ax.plot(lags, y, color=colors[icolor], lw=2)
    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Plot forbidden region
    if (plot_forbidden):
        ax.plot(lags,lags,color='k',lw=2)
        ax.fill_between(lags,lags,0,color='0.8')
    # output
    if not (outfile is None):
        plt.savefig(outfile)
    return ax


def scatter_plot(x, y, marker = '.', color = 'black'):
    p = plt.scatter(x, y, marker = marker, c = color)
    return p


def scatter_matrix(data, marker = '.', color = 'black', size = 1, dim = 6, outfile = None):
    """
    Plots a matrix of scatter plots, where i,j shows data[:,i] vs data[:,j].
     
    dim : the first dim dimensions (at most) are considered
    """
    n = min(dim, len(data[0,:]))
    for i in range (0,n-1):
        for j in range (i+1,n):
            ax = plt.subplot2grid((n-1,n-1), (i, j-1))
            X = data[:,j]
            Y = data[:,i]
            ax.scatter(X, Y, s=size, marker=marker, c= color)
    # output
    if not (outfile is None):
        plt.savefig(outfile)
    return ax
