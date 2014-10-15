#!/usr/bin/env python
# encoding: utf-8
"""
================================================================================
clusters the simulation (MD) data to microstate generators. These generators are
structures or points in state space that shall subsequently be used
to discretize the trajectories using a Voronoi (nearest-neighbor) partition.
================================================================================
\n\n
"""

import sys
import os
import argparse

from pyemma.util.pystallone import JavaException, API
from pyemma.coordinates.clustering import kmeans, regspace
from pyemma.util.files import handleInputFileArg
from pyemma.util.log import getLogger

log = getLogger()
# these are currently supported in pyemma.coordinates.clustering
algorithms = ['kmeans', 'regspace']
metrics = ['minrmsd', 'euclidean']


def handleArgs():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('-i', '--input', nargs='+', dest='input',
                        help='input filename or filename pattern', required=True)
    parser.add_argument('-iformat', help='format of input files (deprecated)',
                        choices=[], default='auto')
    parser.add_argument('-istepwidth', type=int, default=1)
    parser.add_argument('-algorithm', choices=algorithms, required=True)
    parser.add_argument('-metric', choices=metrics, default='euclidian')
    parser.add_argument('-o', '--output', dest='output', default='centers.dat')
    parser.add_argument('--overwrite', action='store_true',
                        help='overwrite existing output files.')
    # algorithm dependent arguments
    parser.add_argument('-clustercenters', '-k', dest='k', type=int)
    parser.add_argument('-dmin', type=float)
    parser.add_argument('-spacing', type=int)
    parser.add_argument('-maxiterations', type=int, default=1000)
    parser.add_argument('-timecolumn', action="store_true",
                        help='this is happily ignored at the moment.')

    args = parser.parse_args()

    # do some sanity checks on input
    if args.algorithm == 'kmeans':
        if args.k is None:
            parser.error('kmeans need parameter -clustercenters')

        if args.k <= 0:
            parser.error('k has to be positive.')

    if args.algorithm == 'regspace' and args.dmin is None:
        parser.error('regularspatial needs parameter -dmin')

#     if args.algorithm == 'regulartemporal' and args.spacing is None:
#         parser.error('regulartemporal needs parameter -spacing')

    # check for input file pattern and create a proper list then
    args.input = handleInputFileArg(args.input)
    if args.input == []:
        log.error('empty input file list! eg. check your pattern and if the files exists.')
        sys.exit(-1)

    # check output file
    if os.path.isfile(args.output) and not args.overwrite:
        log.error('desired output file already exists: "%s"' % args.output)
        sys.exit(-1)
    else:
        try:
            with open(args.output, "w"):
                pass
        except IOError:
            parser.error("output file can not be written: '%s'" % args.output)
    return args


def main():
    args = handleArgs()
    log.info(args)

    try:
        if args.algorithm == 'kmeans':
            clustering = kmeans(args.input, args.k, args.maxiterations)
        elif args.algorithm == 'regspace':
            clustering = regspace(args.input, args.dmin, args.metric)

        # save cluster centers to file
        writer = API.dataNew.writerASCII(args.output, ' ', '\n')
        writer.addAll(clustering._jclustering.getClusterCenters())
        writer.close()
    except JavaException as je:
        log.exception('Java exception occured! message: %s\nstacktrace:\n%s'
                      % (je.message(), je.stacktrace()))
        return 1
    except Exception:
        log.exception('something went wrong...')
        return 1
    log.info("successfully written cluster centers to file %s" % args.output)
    return 0

if __name__ == '__main__':
    sys.exit(main())
