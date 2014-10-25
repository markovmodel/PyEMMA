#!/usr/bin/env python
'''
Created on 13.02.2014

mm_transitionmatrixAnalysis
===========================
calculates various quantities from the transition matrix: Eigenvectors, 
eigenvalues, the stationary distribution, and free energy differences.

@author: marscher
'''
from __future__ import print_function

import sys
import argparse

import numpy as np

from pyemma.msm.io import read_matrix, write_matrix
from pyemma.msm.analysis import is_transition_matrix
from pyemma.util.log import getLogger
from pyemma.msm.analysis.api import stationary_distribution, rdl_decomposition,\
    eigenvectors

log = getLogger()


def handleArgs():
    """ returns parsed arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-inputtransitionmatrix', '-T', dest='T', required=True)
    group = parser.add_argument_group('Free Energy')
    group.add_argument('-freeenergy', dest='freeEnergy',
                       help='output filename of free energy')
    # this is optional if freeenergy is set only
    group.add_argument('-ktfactor', type=float, default=1)

    parser.add_argument('-stationarydistribution',
                        help='output file name of stationary distribution')
    parser.add_argument('-nev', type=int, help='number of eigenvalues')
    parser.add_argument('-eigenvalues', help='output filename of eigenvalues')
    parser.add_argument(
        '-lefteigenvectors', help='output filename for left eigenvectors')
    parser.add_argument(
        '-righteigenvectors', help='output filename for right eigenvectors')

    args = parser.parse_args()

    return args


def freeEnergy(kbTScale, pi):
    """
        Calculates the free energy according to following formula 

        F_{i}=-k_{B}T\:\ln\frac{\pi_{i}}{\max\pi_{j}} 
    """
    freeEnergy = np.empty(pi.shape)
    max_pi = np.max(pi)
    freeEnergy = -kbTScale * np.log(pi / max_pi)
    return freeEnergy


def main():
    args = handleArgs()

    try:
        T = read_matrix(args.T)
    except Exception:
        log.exception('Error occurred during reading of transition matrix')
        return 1

    if not is_transition_matrix(T):
        log.error('given matrix from file "%s" is not a valid transition matrix'
                  % args.inputT)
        return 1

    nev = T.shape[0]  # number of rows
    nevRequested = nev  # if nothing is specified, do the full diagonalization

    if args.nev:  # number of eigenvalues requested
        nevRequested = args.nev
        if nevRequested > nev:
            log.warning("Requesting " + args.nev + "eigenvalues, " +
                        "but transition matrix has only dimension " +
                        T.shape[0] + "x" + T.shape[0] + ".")
            nevRequested = nev

        elif nevRequested <= 0:
            nevRequested = 1

    if args.stationarydistribution or args.freeEnergy:
        calculateStationaryDistribution = True
        if nevRequested <= 1:  # we need only one eigenvalue
            nevRequested = 1
    else:
        calculateStationaryDistribution = False

    w = L = R = None

    if args.lefteigenvectors and args.righteigenvectors:
        # this calculates left and right evs
        log.debug('calc everything')
        log.info('calculating...')
        w, L, R = rdl_decomposition(T, k=nevRequested)
        log.info('calculation finished.')
    elif not args.lefteigenvectors and args.righteigenvectors:
        log.debug('calc only right evs')
        # only right ones?
        R = eigenvectors(T, k=nevRequested, right=True)
    elif not args.righteigenvectors and args.lefteigenvectors:
        log.debug('calc only left evs')
        L = eigenvectors(T, k=nevRequested, right=False)
    else:
        log.warning('Nothing to calculate')
        return 1

    try:
        if L is not None:
            write_matrix(args.lefteigenvectors, L)
        if R is not None:
            write_matrix(args.righteigenvectors, R)
        if w is not None:
            if args.eigenvalues:
                write_matrix(args.eigenvalues, w)
            else:
                print('%i Eigenvalues\n' % nevRequested, w)
    except Exception:
        log.exception(
            'something went wrong during writing left-, rightvectors or eigenvalues')
        return 1

    if calculateStationaryDistribution:
        pi = stationary_distribution(T)
        # safe pi
        if args.statdist_output:
            try:
                write_matrix(args.statdist_output, pi)
            except Exception:
                log.exception('Error during writing stationary distribution to file "%s":'
                              % args.statdist_output)
                return 1
        else:
            print('Stationary Distribution:\n', pi)

    if args.freeEnergy:
        E = freeEnergy(args.ktfactor, pi)
        try:
            write_matrix(args.freeEnergy, E)
        except Exception:
            log.exception('Error during writing free energy to file "%s"'
                          % args.freeEnergy)
            return 1
# TODO: howto decide to print or save to file here?
#     else:
#         print('Free Energy:\n', E)
#
    return 0


if __name__ == '__main__':
    sys.exit(main())
