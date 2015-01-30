#!/usr/bin/env python

__version__ = '2013-01-28 jan'
__author__ = 'Fabian Paul'
__author_email__ = 'fabian.paul@mpikg.mpg.de'

import argparse
import numpy
import sys

def log_loop(iterable):
    for i, x in enumerate(iterable):
        sys.stdout.write('%05d\r' % i)
        sys.stdout.flush()
        yield x

def acf(trajs, acffname, stride=1, max_lag=None, subtract_mean=True, normalize=True, mean=None):
    if subtract_mean and mean == None:
        # compute mean over all trajectories
        from pyemma.coordinates.transform import cocovar
        stats = {}
        print 'computing mean'
        for fname in log_loop(trajs):
            cocovar.run(fname, stats, True, False, False, False, 0)
        mean = stats['mean'] / stats['samples']
 
    acf = numpy.array([[]])
    # number of samples for every tau
    N = numpy.array([])
    
    print 'computing acfs'  
    for fname in log_loop(trajs):
        data = numpy.loadtxt(fname)[::stride]
    if subtract_mean:
        data -= mean
    # calc acfs
    l = data.shape[0]
    fft = numpy.fft.fft(data, n=2 ** int(numpy.ceil(numpy.log2(l * 2 - 1))), axis=0)
    acftraj = numpy.fft.ifft(fft * numpy.conjugate(fft), axis=0).real
    # throw away acf data for long lag times (and negative lag times)
    if max_lag and max_lag < l:
        acftraj = acftraj[:max_lag, :]
    else:
        acftraj = acftraj[:l, :]
        if max_lag:
            sys.stderr.write('Warning: trajectory %s is shorter than maximum lag.' % fname)
    # find number of samples used for every lag 
    Ntraj = numpy.linspace(l, l - acftraj.shape[0] + 1, acftraj.shape[0])
    # adapt shape of acf: resize temporal dimension, additionally set 
    # number of order parameters of acf in the first step
    if acf.shape[1] < acftraj.shape[1] and acf.shape[1] > 0:
        sys.stderr.write('Warning: number of order parameters in %f differs \
                        from the number found in previous trajectories.' % fname)
    if acf.shape[1] < acftraj.shape[1] or acf.shape[0] < acftraj.shape[0]:
        acf.resize(acftraj.shape)
        N.resize(acftraj.shape[0])
    # update acf and number of samples
    acf[0:acftraj.shape[0], :] += acftraj
    N[0:acftraj.shape[0]] += Ntraj
    
    # divide by number of samples
    acf = numpy.transpose(numpy.transpose(acf) / N)
    
    # normalize acfs
    if normalize:
        acf /= acf[0, :].copy()
    
    numpy.savetxt(acffname, acf)
  
def main():
    parser = argparse.ArgumentParser(description='Fast computation of autocorrelation functions over multiple trajectories')
    parser.add_argument('-i', '--trajectories', required=True, nargs='+', help='(input) trajectories', metavar='files')
    parser.add_argument('-o', '--acf', required=True, help='(output) file name for acfs', metavar='file')
    parser.add_argument('-l', '--maxlag', default=None, help='compute acfs upto this lag time', metavar='frames', type=int)
    parser.add_argument('-m', '--meanfree', default=False, help='input data is already mean free, don\'t subtract mean', action='store_true')
    parser.add_argument('-n', '--nonormalize', default=False, help='input trajectories have already variance=1 (don\'t normalize)', action='store_true')
    parser.add_argument('-s', '--stride', default=1, help='only use every n\'th frame', type=int, metavar='n')
    parser.add_argument('-M', '--mean', default=None, help='(input) expert option: file name for mean', metavar='file', type=file)

    args = parser.parse_args()  

    if args.mean:
        mean = numpy.loadtxt(args.mean)
    else:
        mean = None
    
    acf(args.trajectories, args.acf, args.stride, args.maxlag, not args.meanfree, not args.nonormalize, mean)
    return 0

if __name__ == "__main__":
    sys.exit(main())