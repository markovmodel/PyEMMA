#!/usr/bin/env python
'''
Created on 17.02.2014

Estimation of transition matrices
@author: marscher
'''
import argparse
import sys

from pyemma.msm.estimation.api import count_matrix, transition_matrix
from pyemma.msm.io import write_matrix_ascii, load_matrix
from pyemma.util.files import read_dtrajs_from_pattern
from pyemma.util.log import getLogger

log = getLogger()


def handleArgs():
    """ returns parsed arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', dest='discTraj', nargs='+', help='list of discrete trajectories')
    parser.add_argument('-restrictToStates', help='not impled yet')
    parser.add_argument('-reversible', action='store_true', default=False)
    parser.add_argument('-sampling', choices=['slidingwindow', 'lag'])
    parser.add_argument(
        '-stationaryDist', help='estimate transition matrix given a stationary distribution')
    # TODO: if priors are impled, remove this warning
    parser.add_argument(
        '-prior', type=float, default=0.01, help='WARNING: this currently ignored')
    parser.add_argument('-lagtime', type=int, default=1)
    parser.add_argument('-outputtransitionmatrix')
    parser.add_argument('-outputcountmatrix')

    return parser.parse_args()


def main():
    args = handleArgs()

    if args.discTraj != []:
        dtrajs = read_dtrajs_from_pattern(args.discTraj, log)
    else:
        raise ValueError('no valid input given')

    sliding = args.sampling == 'slidingwindow'

    log.info('estimating count matrix with lagtime %i' % args.lagtime)
    log.info('using sliding window approach: %s' % sliding)

    cmatrix = count_matrix(dtrajs, args.lagtime, sliding)

    # TODO: connected cmatrix?
    # largest connected cmatrix?

    # handle prior

    # TODO: impl restrict to states
    # this should be a list of integers
    # how to coarse count_matrix?

    # given a stationary distribution?
    mu = None
    if args.stationaryDist:
        try:
            mu = load_matrix(args.stationaryDist)
        except:
            log.error('error during load of stationary distribution "%s"'
                       % args.stationaryDist)
            log.info('')

    log.info('estimating transition %s matrix...' %
             'reversible' if args.reversible else 'non reversible')

    T = transition_matrix(cmatrix, args.reversible, mu)

    log.info('...finished')

    if args.outputcountmatrix:
        log.info('write count matrix to %s' % args.outputcountmatrix)
        try:
            write_matrix_ascii(args.outputcountmatrix, cmatrix)
        except Exception as e:
            log.exception('Exception during writing of count matrix', e)
            return 1

    if args.outputtransitionmatrix:
        log.info('write transition matrix to %s ' %
                 args.outputtransitionmatrix)
        try:
            write_matrix_ascii(args.outputtransitionmatrix, T)
        except Exception:
            log.exception('Exception during writing of transition matrix')
            return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
