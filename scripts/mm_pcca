#!/usr/local/bin/python2.7
# encoding: utf-8
'''
scripts.mm_pcca -- shortdesc

scripts.mm_pcca is a description

@author:     Martin Scherer
'''
import os
import sys
import argparse

from pyemma.msm.io import read_matrix, write_matrix
from pyemma.msm.analysis import is_transition_matrix, pcca
from pyemma.util.log import getLogger

log = getLogger()

def handleArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputtransitionmatrix', help='input file name of transition matrix' , dest='inputT', required=True)
    #parser.add_argument('-inputeigenvectors', help='optional input filename of eigenvectors of transition matrix', dest="inputEigVecs")
    parser.add_argument('-nclusters', help='desired amount of clusters of microstates', default=2, type=int)
    #parser.add_argument('-ofuzzy', help='output filename of fuzzy')
    parser.add_argument('-ocrisp', help='output filename of crisp')
    parser.add_argument('-osets', help='output filename of sets')
    
    args = parser.parse_args()
    # TODO:; error checking
    if not os.path.exists(args.inputT):
        raise ValueError('given input transition matrix does not exists!')
    
    if args.inputEigVecs != None:
        if os.path.exists(args.inputEigVecs):
            raise ValueError('given input eigenvectors does not exists!')
    return args

def main():
    args = handleArgs()
    
    # read transition matrix
    T = read_matrix(args.inputT)
    if not is_transition_matrix(T):
        raise ValueError('given transition matrix is not a valid transition matrix')
        
    #if args.inputEigVecs:
    #    eig_vecs = read_matrix(args.inputEigVecs)
    memberships = pcca(T, args.nclusters)
    print memberships
    
    # write memberships to file crisp
    try:
        write_matrix(args.ocrisp, memberships)
    except Exception:
        log.exception('error during writing of PCCA memberships.')
        return 1
    
    # TODO: add fuzzy assignment to clusters if supported by pyemma pcca impl
    
    # TODO: add output sets
    
    return 0
    
if __name__ == "__main__":
    sys.exit(main())