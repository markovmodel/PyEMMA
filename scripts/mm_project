#!/usr/bin/env python
'''
Created on 19.11.2013

@author: Fabian Paul <fabian.paul@mpikg.mpg.de>
@author: Martin Scherer
'''

import argparse
import sys

from pyemma.coordinates.transform.tica import Amuse, rename, log_loop

def handleArgs():
    parser = argparse.ArgumentParser(description='Subtract mean and project using a matrix.')

    parser.add_argument('-i', '--trajectories', required=True, nargs='+', help='(input) trajectories', metavar='files')
    parser.add_argument('-T', '--timecolumn', default=False, help='ascii files contain time column?', action='store_true')
    parser.add_argument('-p', '--outdir', required=True, help='(output) directory for order parameter trajectories', metavar='dir')
    parser.add_argument('-W', '--weights', required=True, help='(input) file name for transformation matrix', type=file)
    parser.add_argument('-m', '--mean', required=True, help='(input) file name for data mean', type=file)
    parser.add_argument('-k', '--keepnp', default=None, help='number of dominant order parameters to keep (default=keep all)', metavar='N', type=int)
    parser.add_argument('-d', '--descriptor', default=None, help='string to insert between the file name and the extension of order parameter trajectories', metavar='string')
  
    args = parser.parse_args()
    return args

def main():
    args = handleArgs()
    amuse = Amuse.fromfiles(args.mean, None, args.weights, args.timecolumn)
    
    for t in log_loop(args.trajectories):
        amuse.project(t, rename(t, args.outdir, args.descriptor), amuse.tica_weights, args.keepnp)


if __name__ == '__main__':
    sys.exit(main())