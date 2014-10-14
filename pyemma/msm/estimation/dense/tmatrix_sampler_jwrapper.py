'''
Created on Jun 6, 2014

@author: marscher
'''
from pyemma.util.log import getLogger
from pyemma.util.pystallone import ndarray_to_stallone_array,\
            JavaException, stallone_array_to_ndarray, stallone

import numpy as np
from scipy.sparse.base import issparse

__all__ = ['ITransitionMatrixSampler']

class ITransitionMatrixSampler(object):
    """
    samples a (reversible) transition matrix from given count matrix and optional given 
    stationary distribution.
    """
    
    def __init__(self, counts, mu = None, reversible=False, Tinit = None):
        """
        Sets the count matrix used for sampling. Assumes that the prior 
        (if desired) is included.
       
        Parameters
        ----------
        counts : ndarray (n, n)
            the posterior count matrix
        mu : ndarray (n)
           optional stationary distribution, if given, the sampled transition matrix
           will have this this stat dist.
        reversible : boolean
           should sample a reversible transition matrix.
           
        Tinit : ndarray(n, n)
           optional start point for sampling algorithm.
           
        Example
        -------
        >>> C = np.array([[5, 2], [1,10]]) 
        >>> sampler = ITransitionMatrixSampler(C)
        >>> T = sampler.sample(10**6)
        >>> print T
        
        """
        if issparse(counts):
            counts = counts.toarray()
        # the interface in stallone takes counts as doubles
        counts = counts.astype(np.float64)
        
        try:
            C = ndarray_to_stallone_array(counts)
            jpackage = stallone.mc.sampling
            # convert types to java
            if Tinit is not None:
                Tinit = ndarray_to_stallone_array(Tinit)
            if mu is not None:
                mu = ndarray_to_stallone_array(mu)
                
            if reversible:
                if mu: # fixed pi
                    if Tinit:
                        self.sampler = jpackage.TransitionMatrixSamplerRevFixPi(C, Tinit, mu)
                    else:
                        self.sampler = jpackage.TransitionMatrixSamplerRevFixPi(C, mu)
                else: # sample reversible matrix, with arbitrary pi
                    if Tinit:
                        self.sampler = jpackage.TransitionMatrixSamplerRev(C, Tinit)
                    else:
                        self.sampler = jpackage.TransitionMatrixSamplerRev(C)
            else: # sample non rev
                if Tinit:
                    self.sampler = jpackage.TransitionMatrixSamplerNonrev(C, Tinit)
                else:
                    self.sampler = jpackage.TransitionMatrixSamplerNonrev(C)
                
        except JavaException as je:
            log = getLogger()
            log.exception("Error during creation of tmatrix sampling wrapper:"
                          " stack\n%s" %je.stacktrace())
            raise
    
    def sample(self, steps):
        """
        Generates a new sample
        Parameters
        ----------
        steps : int
        the number of sampling steps taken before the next sample is returned
        
        Returns
        -------
        sample : ndarray
        """
        return stallone_array_to_ndarray(self.sampler.sample(steps))
    
    def loglikelihood(self):
        """
        Returns
        -------
        the log-likelihood of the current sample : float
        """
        return self.sampler.logLikelihood()
        
