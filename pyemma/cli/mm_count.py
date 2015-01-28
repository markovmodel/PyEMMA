#!/usr/bin/env python
'''
mm_count
Counts transitions in discrete trajectories (-i) for a given set of microstates (-s).

Created on 31.03.2014

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
#      -i (<filename|filenamepattern>)+
#  -o <filename>
#  -s <filename>
    
    parser.add_argument('-i', dest='discTrajs', required=True, nargs='+', help='list of discrete trajectories or pattern.')
    parser.add_argument('-o', dest='output', required=True, help='output filename of largest connected set')
    parser.add_argument('-s', dest='microstates', required=True, help='input filename of microstates to count transitions for')

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