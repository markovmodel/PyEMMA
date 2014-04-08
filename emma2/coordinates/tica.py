r"""
====
TICA
====

TODO: describe usage here

Created on 19.11.2013
@author: Fabian Paul <fabian.paul@mpikg.mpg.de>
@author: marscher

"""

import os
import sys
import numpy

__docformat__ = "restructuredtext en"
__all__ = ['correlation', 'log_loop', 'rename', 'Amuse']

''' import correlation covariance C extension module '''
import cocovar
from emma2.util.log import getLogger
log = getLogger()

def correlation(cov, var):
    '''Calculate covariance matrix from correlation matrix.'''
    
    n = cov.shape[0]
    corr = numpy.empty([n, n])
    for i in xrange(n):
        for j in xrange(n):
            corr[i, j] = cov[i, j] / numpy.sqrt(var[i] * var[j])
    return corr

def log_loop(iterable):
    '''Loop over iterable and display counter on stdout.'''
  
    for i, x in enumerate(iterable):
        sys.stdout.write('%05d\r' % i)
        sys.stdout.flush()
        yield x

def rename(fname, directory, inset=None):
    '''Change directory and extension of file name. 
    New extension is inset+original extension.'''

    res = directory + os.sep + os.path.basename(fname)
    if inset:
        name, orig_ext = os.path.splitext(res)
        res = name + inset + orig_ext
    return res

class Amuse:
    """
    TODO: document class
    """

    @classmethod
    def fromfiles(cls, mean, pca_weights, tica_weights, time_column=False):
        '''Initialize from files.'''

        amuse = cls(time_column)
        amuse.mean = numpy.genfromtxt(mean)
        error = Exception('Number of dimensions in mean and matrix does not agree.')
        if pca_weights:
            amuse.pca_weights = numpy.genfromtxt(pca_weights)
            if not amuse.pca_weights.shape[0] == amuse.pca_weights.shape[1] == amuse.mean.shape[0]:
                raise error
        if tica_weights:
            amuse.tica_weights = numpy.genfromtxt(tica_weights)
            if not amuse.tica_weights.shape[0] == amuse.tica_weights.shape[1] == amuse.mean.shape[0]:
                raise error
        amuse.n = amuse.mean.shape[0]
        return amuse
    
    @classmethod  
    def compute(cls, files, lag, normalize=False, time_column=False, mean=None):
        """
        compute it from given files
        """
        amuse = cls(time_column)
        if not files:
            raise Exception('No input trajectories were given.')

        # calculate mean
        if mean == None:
            log.info('computing mean')
            mean_stats = {}
            for f in log_loop(files):
                cocovar.run(f, mean_stats, True, False, False, False, 0)
            amuse.mean = mean_stats['mean'] / mean_stats['samples']
        else:
            # use mean specified by the user
            if amuse.time_column:
                amuse.mean = numpy.hstack((numpy.zeros(1), mean))
            else:
                amuse.mean = mean

        # calculate rest of statistics
        log.info('computing covariances')
        stats = { 'mean': amuse.mean }
        for f in log_loop(files):
            cocovar.run(f, stats, False, False, True, True, lag)
        
        amuse.n = stats['cov'].shape[0]
 
        cov = numpy.array(stats['cov'] / stats['samples'], dtype=float)
        tcov = numpy.array(stats['tcov'] / stats['tcov_samples'], dtype=float)
      
        amuse.var = cov.diagonal()
  
        if normalize:
            corr = correlation(cov, amuse.var)
            tcorr = correlation(tcov, amuse.var)
        else:
            corr = cov
            tcorr = tcov
    
        # symmetrization
        tcorr = 0.5 * (tcorr + numpy.transpose(tcorr))
             
        # remove time column
        if amuse.time_column:
            corr = corr[1:, 1:]
            tcorr = tcorr[1:, 1:]
            amuse.var = amuse.var[1:]
            amuse.mean = amuse.mean[1:]
            amuse.n = amuse.n - 1
    
        amuse.pca_values, amuse.pca_weights = numpy.linalg.eig(corr)
        # normalize weights by dividing by the standard deviation of the pcs 
        for i, l in enumerate(amuse.pca_values):
            amuse.pca_weights[:, i] = amuse.pca_weights[:, i] / numpy.sqrt(l)

        pc_tcorr = numpy.dot(numpy.dot(numpy.transpose(amuse.pca_weights), tcorr), amuse.pca_weights)
        amuse.tica_values, amuse.intermediate_weights = numpy.linalg.eig(pc_tcorr)
        amuse.tica_weights = numpy.dot(amuse.pca_weights, amuse.intermediate_weights)

        # sort eigenvalues und eigenvectors
        sort_perm = amuse.pca_values.argsort()[::-1]
        amuse.pca_values = amuse.pca_values[sort_perm]
        amuse.pca_weights = amuse.pca_weights[:, sort_perm]

        sort_perm = amuse.tica_values.argsort()[::-1]
        amuse.tica_values = amuse.tica_values[sort_perm]
        amuse.tica_weights = amuse.tica_weights[:, sort_perm]

        # if the transformation involves scaling of the original coordinates
        # using standard deviation, include this transformation in the matrices
        if normalize:
            std = numpy.sqrt(amuse.var)
            amuse.tica_weights = numpy.transpose(numpy.transpose(amuse.tica_weights) / std)
            amuse.pca_weights = numpy.transpose(numpy.transpose(amuse.pca_weights) / std)
            
        amuse.corr = corr
        amuse.tcorr = tcorr

        return amuse

    def __init__(self, time_column):
        """
        TODO: document this
        """
        self.time_column = time_column
    
    def pca(self, fin, fout, keep_pc):
        '''Perform PCA on data'''
        self.project(fin, fout, self.pca_weights, keep_pc)

    def tica(self, fin, fout, keep_ic):
        '''Perform TICA on data'''
        self.project(fin, fout, self.tica_weights, keep_ic)
    
    def project(self, fin, fout, weights, keep_n=None):
        """
        project it
        """
        if fin == fout:
            raise Exception('Input file name is equal to output file name.')
        if not keep_n:
            keep_n = self.n  # keep all
        cocovar.project(fin, fout, self.mean, weights, keep_n, self.time_column)
