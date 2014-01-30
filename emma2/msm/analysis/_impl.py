'''
Created on 18.10.2013

@author: marscher
'''
__all__ = ['TPTFlux']

from emma2.util.log import getLogger
log = getLogger()

from emma2.util.pystallone import API, JavaException, \
    stallone_array_to_ndarray, ndarray_to_stallone_array

class TPTFlux():
    """
        This class wraps around a stallone ITPTFlux class.
    """
    def __init__(self, T, A, B):
        """
        initializes this with given matrix
        Parameters
        ----------
        T : Transition matrix
            ndarray
        A : set of states
            ndarray( dtype = int)
        B : set of states 
            ndarray( dtype = int)
        Throws
        ------
        RuntimeError, if stallone is not available
        """
        import numpy as np
        try:
            T = T.astype(np.float64)
            A = np.asarray(A).astype(np.int64)
            B = np.asarray(B).astype(np.int64)
            A = ndarray_to_stallone_array(A)
            B = ndarray_to_stallone_array(B)
            T = ndarray_to_stallone_array(T)
            log.debug('creating TPTFlux instance and calculate fluxes...')
            self.ITPT = API.msmNew.createTPT(T, A, B)
            log.debug('finished TPTFlux calculation.')
        except JavaException as je:
            msg = str(je.message()) + '\n' + \
                str(je.stacktrace().replace(',', '\n'))
            log.error(msg)
            raise
        except TypeError as t:
            log.error("type error occurred: %s" %t)
            raise
        except Exception as e:
            log.error("unknown error occurred: %s" %e)
            raise
    
    def getBackwardCommittor(self):
        """
        Returns
        -------
        Backward Committor : ndarray
        """
        return stallone_array_to_ndarray(self.ITPT.getBackwardCommittor())
    
    def getFlux(self):
        """
        Returns
        -------
        Flux : ndarray
        """
        return stallone_array_to_ndarray(self.ITPT.getFlux())
    
    def getForwardCommittor(self):
        """
        Returns
        -------
        Forward Committor : ndarray
        
        """
        return stallone_array_to_ndarray(self.ITPT.getForwardCommittor())
    
    def getNetFlux(self):
        """
        Returns
        -------
        Net flux : ndarray
        
        """
        return stallone_array_to_ndarray(self.ITPT.getNetFlux())
    
    def getRate(self):
        """
        Returns
        -------
        Rate : float        
        """
        return self.ITPT.getRate()
    
    def getStationaryDistribution(self):
        """
        Returns
        -------
        Stationary distribution : ndarray
        
        """
        return stallone_array_to_ndarray(self.ITPT.getStationaryDistribution())
    
    def getTotalFlux(self):
        """
        Returns
        -------
        Total flux : float
        
        """
        return self.ITPT.getTotalFlux()
        
    def setRateMatrix(self, K):
        """
        Parameters
        ----------
        K (Rate matrix) : ndarray
        
        """
        self.K = ndarray_to_stallone_array(K)
        
    def setStationaryDistribution(self, pi):
        """
        Parameters
        ----------
        pi : stationary distribution
        
        """
        self.pi = ndarray_to_stallone_array(pi)
        
    def setTransitionMatrix(self, T):
        """
        Parameters
        ----------
        T : ndarray
        """
        self.T = ndarray_to_stallone_array(T)
