#!/usr/bin/env python
'''
Created on 29.11.2013

@author: marscher
'''
import sys
import argparse

from pyemma.msm.io.api import read_matrix, write_matrix
from pyemma.util.log import getLogger
from pyemma.msm.analysis.api import tpt, is_transition_matrix

log = getLogger()

def handleArgs():
    """ returns parsed arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputtransitionmatrix', dest='T', required=True)
    parser.add_argument('-statdist', help='optional stationary distribution')
    parser.add_argument('-seta')
    parser.add_argument('-setb')
    # output tpt
    parser.add_argument('-oforwardcommittor')
    parser.add_argument('-obackwardcommittor')
    parser.add_argument('-oflux')
    parser.add_argument('-onetflux')
    
    # coarse sets
    # TODO: impl
#     parser.add_argument('-coarsesets')
#     parser.add_argument('-ocoarseforwardcommittor')
#     parser.add_argument('-ocoarsebackwardcommittor')
#     parser.add_argument('-ocoarseflux')
#     parser.add_argument('-ocoarsenetflux')
    
    return parser.parse_args()

def main():
    args = handleArgs()
    
    try:
        T = read_matrix(args.T)
        if not is_transition_matrix(T):
            raise ValueError
    except Exception as e:
        log.exception('Error occurred during reading of transition matrix.', e)
        return 1
    
    try:
        A = read_matrix(args.seta)
    except Exception as e:
        log.exception('Error occurred during reading of set A.', e)
        return 1
    
    try:
        B = read_matrix(args.setb)
    except Exception as e:
        log.exception('Error occurred during reading of set B.', e)
        return 1
    
    log.debug('create tpt wrapper object.')
    TPTFlux = tpt(T, A, B)
    log.debug('tpt wrapper object created.')
    log.debug('converting output from java to python.')
    flux = TPTFlux.getFlux()
    netFlux = TPTFlux.getNetFlux()
    forwardCommittor = TPTFlux.getForwardCommittor()
    backwardCommittor = TPTFlux.getBackwardCommittor()
    log.debug('successfully converted.')
    
    log.info("Forward committor is: ", forwardCommittor)
    log.info("Backward committor is: " + backwardCommittor)
    
    log.info("Total flux    : ", flux)
    log.info("Net flux      : ", netFlux)
    
    # write results if wanted
    if args.oforwardcommittor:
        try:
            write_matrix(args.oforwardcommittor, forwardCommittor)
        except Exception as e:
            log.exception('Error occurred during writing of forward committor.', e)
            return 1
        
    if args.obackwardcommittor:
        try:
            write_matrix(args.obackcommittor, backwardCommittor)
        except Exception as e:
            log.exception('Error occurred during writing of backward committor.', e)
            return 1
        
    if args.oflux:
        try:
            write_matrix(args.oflux, flux)
        except Exception as e:
            log.exception('Error occurred during writing of flux.', e)
            return 1

    if args.onetflux:
        try:
            write_matrix(args.onetflux, netFlux)
        except Exception as e:
            log.exception('Error occurred during writing of net flux.', e)
            return 1

if __name__ == '__main__':
    sys.exit(main())