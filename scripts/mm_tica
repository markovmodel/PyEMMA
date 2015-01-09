#!/usr/bin/env python

__version__ = '2013-01-28 jan'
__author__ = 'Fabian Paul'
__author_email__ = 'fabian.paul@mpikg.mpg.de'

import argparse
import numpy
import sys

from pyemma.coordinates.transform.tica import Amuse

def handleArgs():
    parser = argparse.ArgumentParser(description='Calculate covariance matrices and weight matrices for PCA and TICA.')

    parser.add_argument('-i', '--trajectories', required=True, nargs='+', help='(input) trajectories, either in ascii or dcd format', metavar='files')
    parser.add_argument('-T', '--timecolumn', default=False, help='ascii files contain time column?', action='store_true')
    parser.add_argument('-n', '--normalize', default=False, help='use correlation matrices instead of covariance matrices', action='store_true')
    parser.add_argument('-l', '--lagtime', help='lag time parameter tau for TICA (in frames)', metavar='frames', type=int)
    parser.add_argument('-c', '--covariancematrix', help='(output) file name for covariance matrix', metavar='file')
    parser.add_argument('-t', '--laggedcovariancematrix', help='(output) file name for symmetric, time-lagged covariance matrix', metavar='file')
    parser.add_argument('-C', '--covariances', help='(output) file name for PC eigenvalues (PC variances)', metavar='file')
    parser.add_argument('-S', '--timescales', help='(output) file name for IC eigenvalues (lagged IC variances)', metavar='file')
    parser.add_argument('-W', '--pcaweights', help='(output) file name for scaled PCA eigenvectors WSigma^-1', metavar='file')
    parser.add_argument('-V', '--ticaweights', help='(output) file name for TICA eigenvectors WSigma^-1V', metavar='file')
    parser.add_argument('-m', '--mean', help='(output) file name for data mean', metavar='file')
    parser.add_argument('-v', '--var', help='(output) file name for data variance', metavar='file')
    parser.add_argument('-M', '--usemean', help='(input) expert option: file name for mean', metavar='file', type=file)

    args = parser.parse_args()
    return args

def main():
    args = handleArgs()

    # --lagtime is only required for tica, handle this
    if not args.lagtime:
        if args.ticaweights or args.laggedcovariancematrix or args.timescales:
            raise Exception('Muse select a value for --lagtime, if --ticaweights/--laggedcovariancematrix/--timescales is wanted.')
        else:
            lagtime = 1
    else:
        lagtime = args.lagtime
    
    if(args.usemean):
        mean = numpy.genfromtxt(args.usemean)
    else:
        mean = None
    
    amuse = Amuse.compute(args.trajectories, lagtime, args.normalize, args.timecolumn, mean)
    
    if(args.covariances): numpy.savetxt(args.covariances, amuse.pca_values)
    if(args.timescales): numpy.savetxt(args.timescales, amuse.tica_values)
    if(args.pcaweights): numpy.savetxt(args.pcaweights, amuse.pca_weights)
    # TODO: only write if tica was really used.
    if(args.ticaweights): numpy.savetxt(args.ticaweights, amuse.tica_weights)
    if(args.mean): numpy.savetxt(args.mean, amuse.mean)
    if(args.var): numpy.savetxt(args.var, amuse.var)
    if(args.covariancematrix): numpy.savetxt(args.covariancematrix, amuse.corr)
    if(args.laggedcovariancematrix): numpy.savetxt(args.laggedcovariancematrix, amuse.tcorr)

    return 0

if __name__ == '__main__':
    sys.exit(main())