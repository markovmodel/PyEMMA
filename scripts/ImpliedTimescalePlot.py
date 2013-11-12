# -*- coding: utf-8 -*-
import emma2.msm.estimation as est
import emma2.msm.analysis as ana
import emma2.msm.io as io

from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
from numpy import ndarray

import logging
log = logging.getLogger('emma2/scripts/LagTimePlot')

import argparse

traj = io.read_dtraj(args.trajectory)
k = args.k

def calculate_impl_timescales(tau):
    """
    Returns tuple (tau, timescales)
    """
    log.info("estimating count matrix for lagtime %i" % tau)
    C = est.cmatrix(traj, lag = tau)
    log.info("estimating transition matrix for lagtime %i" % tau)
    T = est.tmatrix(C, reversible = True)
    log.info("calculating implied time scales lagtime %i" % tau)
    ts = ana.timescales(T, tau, k)
    log.info('estimation finished for lagtime %i' % tau)
    return (tau, ts)

def perform():
    """ create a process pool with cpu workers and calculate 
        transition matrices for range [tau_min, tau_max] with given step size.
    """
    pool = Pool(cpu_count())
    
    """ sort given times and perform calculation"""
    args.lagtimes.sort()
    time_scales = pool.map(calculate_impl_timescales, args.lagtimes)
    
    """ calculate grid for plotting"""
    num_plots = len(time_scales)
    if num_plots >= 2:
        r = num_plots % 2
        num_cols = 2 + r
        num_rows = max(1, num_plots - num_cols)
    else:
        num_cols = num_rows = 1
    
    log.debug("(rows, cols) = (%i, %i)" % (num_rows, num_cols))
    f, axarr = plt.subplots(nrows = num_rows, ncols = num_cols)
    """ wrap axarr as a single element list for zip, if it is not an ndarray""" 
    unpack = lambda x: x.flatten() if isinstance(x, ndarray) else [x]
    
    for ts, ax in zip(time_scales, unpack(axarr)):
        if not isinstance(ts, tuple):
            log.debug("nothing to plot")
            continue
        ax.plot(ts[1], 'ro')
        ax.set_xlabel(r'$\psi_i$ Eigenvalue / Process index')
        ax.set_ylabel('Implied time scale [ns]')
        #ax.set_yscale('log')
        ax.set_title(r'Implied time scales for $\tau = %i$' % ts[0])
        ax.tau = ts[0]
    
    """ remove unused subplots (those which do not have 'tau' attribute """
    for ax in unpack(axarr):
        try:
            ax.tau
        except AttributeError:
            f.delaxes(ax) 
    
    plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0)
    
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == '__main__':
    message='Implied time scales plotting arguments'
    parser = argparse.ArgumentParser(description=message)
    message='filename of discretized trajectory to operate on.'
    parser.add_argument('--trajectory', '-t', help=message)
    parser.add_argument('-k', help='k implied timescales (eigenvalues)',\
                            type=int, default=10)
    message='list of lag time values (integers). Eg. -l 10 100 1000'
    parser.add_argument('--lagtimes', '-l', help=message, 
                    nargs='+', type=int, default=[1])
    parser.add_argument('--output', '-o', help='output file name', default='')
    parser.add_argument('--numprocs', '-p', type=int, default = cpu_count(),
                    help='number of processes used for calculation')
    args = parser.parse_args()
    perform()
