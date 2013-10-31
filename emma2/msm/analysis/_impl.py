'''
Created on 18.10.2013

@author: marscher
'''
import emma2.util.pystallone as stallone
from emma2.util.ArrayWrapper import ArrayWrapper

class TPT():
    """
        This class wraps around a stallone ITPTFlux class.
        Performs internal array wrapping between python and java (bad performance)

    """
    def __init__(self, T, A, B):
        """
        initializes this with given matrix
        Parameters
        ----------
        T : Transition matrix
            ndarray
        A : set of states
        # TODO: shape?
            ndarray( dtype = int, shape=?)
        B : set of states 
            ndarray( dtype = int)
        # TODO: is this valid in sphinx?
        Throws
        ------
        RuntimeError, if stallone is not available
        """
        try:
            A = stallone.ndarray_to_stallone_array(A)
            B = stallone.ndarray_to_stallone_array(B)
            T = stallone.ndarray_to_stallone_array(T)
            self.ITPT = stallone.API.msmNew.createTPT(T, A, B)
            # TODO: consider: should we call calculate here?
        except stallone.JavaError as je:
            exception = je.getJavaException()
            msg = str(exception) + '\n' + \
                str(exception.getStackTrace()).replace(',', '\n')
            raise RuntimeError(msg)
    
    def getBackwardCommittor(self):
        """
        Returns
        -------
        Backward Committor : ndarray
        """
        return ArrayWrapper(self.ITPT.getBackwardCommittor())
    
    def getFlux(self):
        """
        Returns
        -------
        Flux : ndarray
        """
        return ArrayWrapper(self.ITPT.getFlux())
    
    def getForwardCommittor(self):
        """
        Returns
        -------
        Forward Committor : ndarray
        
        """
        return ArrayWrapper(self.ITPT.getForwardCommittor())
    
    def getNetFlux(self):
        """
        Returns
        -------
        Net flux : ndarray
        
        """
        return ArrayWrapper(self.ITPT.getNetFlux())
    
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
        return ArrayWrapper(self.ITPT.getStationaryDistribution())
    
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
        self.K = stallone.ndarray_to_stallone_array(K)
        
    def setStationaryDistribution(self, pi):
        """
        Parameters
        ----------
        pi : stationary distribution
        
        """
        self.pi = stallone.ndarray_to_stallone_array(pi)
        
    def setTransitionMatrix(self, T):
        """
        Parameters
        ----------
        T : ndarray
        """
        self.T = stallone.ndarray_to_stallone_array(T)
