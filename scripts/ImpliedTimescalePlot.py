# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import emma2.msm.estimation as est
import emma2.msm.analysis as ana
import emma2.msm.io as io

import matplotlib.pyplot as plt

import logging
log = logging.getLogger('emma2/scripts/LagTimePlot')

import argparse

parser = argparse.ArgumentParser(description='Implied time scales plotting arguments')

parser.add_argument('--trajectory', '-t',
                    help='filename of discretized trajectory to operate on.')
parser.add_argument('-k', help='k implied timescales (eigenvalues)',
                    type=int, default=10)
parser.add_argument('--lagtimes', '-l',
                    help='list of lag time values (integers). Eg. -l 10 100 1000',
                    nargs='+', type=int, default=[1])
parser.add_argument('--output', '-o', help='output file name', default='')
args = parser.parse_args()

traj = io.read_dtraj(args.trajectory)
k = args.k

def impl_timescales(tau):
    """
    Returns tuple (tau, timescales)
    """
    log.info("estimating count matrix for lagtime %i" % tau)
    try:
        C = est.cmatrix(traj, lag = tau)
        T = est.tmatrix(C, reversible = True)
        return (tau, ana.timescales(T, tau, k))
    except Exception as e:
        print e

def perform():
    """ create a process pool with 4 workers and calculate 
        transition matrices for range [tau_min, tau_max] with given step size.
    """
    from multiprocessing import Pool, cpu_count
    pool = Pool(cpu_count())
    
    range_ = args.lagtimes
    time_scales = pool.map(impl_timescales, range_)
    
    """ calculate grid for plotting"""
    num_plots = len(time_scales)
    r = num_plots % 2
    num_cols = 2 + r
    num_rows = max(1, num_plots - num_cols)
    log.debug("(rows, cols) = (%i, %i)" % (num_rows, num_cols))
    f, axarr = plt.subplots(nrows = num_rows, ncols = num_cols)
    
    for ts, ax in zip(time_scales, axarr.flatten()):
        if not isinstance(ts, tuple):
            print "nothing to plot"
            continue
        ax.plot(ts[1], 'ro')
        ax.set_xlabel(r'$\psi_i$ Eigenvalue')
        ax.set_ylabel('Implied time scale [ns]')
        ax.set_title(r'Implied time scales for $\tau = %i$' % ts[0])
    
    plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 1.0)
    
    if args.output:
        plt.savefig(args.output)
    else:
        plt.show()

if __name__ == '__main__':
    perform()
