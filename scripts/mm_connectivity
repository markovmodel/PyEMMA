#!/usr/bin/env python
'''
Created on 17.02.2014

@author: marscher
'''
import argparse
import sys

from pyemma.msm.estimation.api import count_matrix, largest_connected_set
from pyemma.msm.io.api import write_matrix
from pyemma.util.files import read_dtrajs_from_pattern
from pyemma.util.log import getLogger


log = getLogger()

def handleArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', dest='discTraj', required=True, nargs='+', help='list of discrete trajectories')
    parser.add_argument('-o', dest='output', help='output filename of largest connected set')
    parser.add_argument('-lag', dest='lag', help='lag time for which connectivity should be calculated for', type=int, default=1)
    return parser.parse_args()

def main():
    args = handleArgs()
    
    if args.discTraj != []:
        dtrajs = read_dtrajs_from_pattern(args.discTraj, log)
    else:
        raise ValueError('no valid input given')

    cmatrix = count_matrix(dtrajs, args.lag, sliding=True)
    
    lcs = largest_connected_set(cmatrix)
    if args.output:
        write_matrix(args.output, lcs)
    else:
        print lcs
    return 0
    
if __name__ == '__main__':
    sys.exit(main())