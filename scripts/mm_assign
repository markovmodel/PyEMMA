#!/usr/bin/env python
# encoding: utf-8
"""
================================================================================
assigns the simulation data (-i) to microstates (-ic) that have been previously
generated/ or user supplied. The assignment is done such that each frame of the
input data is assigned to the number of the nearest cluster center (Voronoi parition).
================================================================================
\n\n
"""
import argparse
import os
import sys

from pyemma.util.files import paths_from_patterns
from pyemma.util.log import getLogger
from pyemma.coordinates.clustering import assign

from pyemma.util.pystallone import API, JString, JavaException

log = getLogger()

metrics = ['minrmsd', 'euclidean']


def handleArgs():
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('-i', dest='trajs', required=True, nargs='+',
                        help='list of trajectories to assign to clusters')
    parser.add_argument('-istepwidth', type=int, default=1, help='')
    parser.add_argument('-imetric', dest='metric', choices=metrics, default=metrics[1])
    parser.add_argument('-ic', dest='clusterCenters', required=True,
                        help='previously generated microstates (cluster centers)')
    parser.add_argument('-o', dest='output', default='.',
        help='output directory for discretized trajectories, defaults to current directory')
    parser.add_argument('-timecolumn', action="store_true", help='this is happily ignored at the moment.')

    args = parser.parse_args()

    if not os.path.isdir(args.output):
        parser.error('given output %s is not a directory' % args.output)

    trajs = paths_from_patterns(args.trajs)
    if trajs == []:
        parser.error('given trajectory pattern "%s" does not match any files' % args.trajs)
    args.trajs = trajs

    if not os.access(args.output, os.W_OK):
        raise RuntimeError("Argument given by -o '%s' is not writeable!" % args.output)

    return args


def main():
    args = handleArgs()

    try:
        # create IDataReader of cluster center
        centersFile = API.dataNew.readerASCII(JString(args.clusterCenters))
        centersFile.scan()
        # load to type IDataSequence
        centers = centersFile.load()
    except JavaException:
        log.exception('Error during reading of cluster centers file.')
        return 1

    # create metric and discretization from factory, reflect options
    try:
        dim = centers.dimension()
        clusterNew = API.clusterNew
        if args.metric == 'euclidean':
            jmetric = API.clusterNew.metric(clusterNew.METRIC_EUCLIDEAN, dim)
        elif args.metric == 'minrmsd':
            jmetric = API.clusterNew.metric(clusterNew.METRIC_MINRMSD, dim/3)
        disc = API.discNew.voronoiDiscretization(centers, jmetric)
    except JavaException:
        log.exception('Error during reading of cluster centers file.')
        return 1

    # construct output file names for discrete trajectories
    outfiles = []
    for t in args.trajs:
        base = os.path.basename(t)
        base, ext = os.path.splitext(base)
        outfile = os.path.join(args.output, base + ".dtraj")
        log.debug("map %s -> %s" % (t, outfile))
        outfiles.append(outfile)

    # assign to dtrajs
    try:
        assign(args.trajs, disc, outfiles, return_discretization=False)
    except JavaException as je:
        log.exception('Java exception occured! message: %s\nstacktrace:\n%s'
                      % (je.message(), je.stacktrace()))
        return 1
    except Exception:
        log.exception('error during assignment.')
        return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
