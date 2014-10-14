#!/usr/bin/env python
# encoding: utf-8
from pyemma.coordinates.clustering.api import kmeans, regspace, kregspace, assign
usage = """
================================================================================
clusters the simulation (MD) data to microstate generators. These generators are
structures or points in state space that are directly used 
to discretize the trajectories using a Voronoi (nearest-neighbor) partition.
================================================================================
\n\n
"""

import argparse
import sys
import os

from pyemma.util.files import handleInputFileArg
from pyemma.util.log import getLogger
from pyemma.util.pystallone import JavaException, API

log = getLogger()
# these are currently supported in pyemma.coordinates.clustering
algorithms = ['kmeans', 'regspace', 'kregspace']
metrics = ['rmsd', 'euclidean']

def handleArgs():
    parser = argparse.ArgumentParser(usage=usage)
    parser.add_argument('-i', '--input', nargs='+', dest='input', help='input filename or filename pattern', required=True)
    parser.add_argument('-iformat', help='format of input files (deprecated)',
                         choices=[], default='auto')
    
    parser.add_argument('-istepwidth', type=int, default=1)
    parser.add_argument('-algorithm', choices=algorithms, required=True)
    parser.add_argument('-metric', choices=metrics, default='euclidian')
    parser.add_argument('-oc', '--output-centers', dest='outputCenters',
                         default='centers.dat', help='filename for cluster centers.')
    parser.add_argument('-od', '--output-dtraj', dest='outputDTraj',
                        help='output directory for discrete trajectories (dtraj).'
                        ' If none is given, dtrajs will be written next to their input files')
    
    # cluster algorithm dependent arguments
    parser.add_argument('--clustercenters', '-k', dest='k', type=int)
    parser.add_argument('-dmin', type=float)
    #parser.add_argument('-spacing', type=int)
    parser.add_argument('-maxiterations', type=int, default=100)
    #parser.add_argument('-timecolumn', action="store_true", help='this is happily ignored at the moment.')
    
    args = parser.parse_args()

    # do some sanity checks on input
    if args.algorithm == 'kmeans' or args.algorithm == 'kcenter':
        if args.k is None:
            parser.error('kmeans and kcenter need parameter -clustercenters') 
            
    if args.algorithm == 'regspace' and args.dmin is None:
        parser.error('regularspatial needs parameter -dmin')
        
#     if args.algorithm == 'regulartemporal' and args.spacing is None:
#         parser.error('regulartemporal needs parameter -spacing')
    
    # check for input file pattern and create a proper list then
    args.input = handleInputFileArg(args.input)
    if args.input == []:
        log.error('empty input file list! eg. check your pattern and if the files exists.')
        sys.exit(-1)
        
        
    if args.outputDTraj and not os.path.isdir(args.outputDTraj):
        log.error('value given for discrete trajectories output (%s) is not a valid directory' % args.outputDTraj)
        sys.exit(-1)
        
    return args

def main():
    args = handleArgs()
    
    try:
        if args.algorithm == 'kmeans':
            clustering = kmeans(args.input, args.k, args.maxiterations)
        elif args.algorithm == 'regspace':
            clustering = regspace(args.input, args.dmin, args.metric)
        elif args.algorithm == 'kregspace':
            clustering = kregspace(args.input, args.k)
        
        # construct output file names for discrete trajectories
        outfiles = []
        for t in args.input:
            log.debug('creating dtraj name from "%s"'%t)
            if args.outputDTraj:
                # strip path from traj
                t = os.path.basename(t)
            base, ext = os.path.splitext(t)
            outfiles.append( base + ".dtraj")
        
        log.info('create discrete trajectories: %s' % outfiles)
        assign(args.input, clustering, outfiles, return_discretization=False)
        
    except JavaException as je:
        log.exception('Java exception occured! message: %s\nstacktrace:\n%s' 
                % (je.message(), je.stacktrace()))
        return 1
    except Exception:
        log.exception('something went wrong...')
        return 1
    return 0

if __name__ == '__main__':
    sys.exit(main())
