'''
Created on 18.10.2013

@author: marscher
'''
import numpy as np
from decimal import Decimal # for wrapping java.math.BigDecimal

from emma2.util.log import getLogger
from emma2.util.pystallone import API, JavaException, \
    stallone_array_to_ndarray, ndarray_to_stallone_array, stallone, \
    JArray, JInt, JDouble

__all__ = ['TPTFlux', 'PathwayDecomposition']

log = getLogger()

class TPTFlux(object):
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
            log.exception("type error occurred.", t)
            raise
        except Exception as e:
            log.exception("unknown error occurred.", e)
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
        
# TODO: test
class PathwayDecomposition(object):
    
    def __init__(self, F, Q, A, B):
        """
        Parameters
        ----------
        F : The net fluxes matrix
          ndarray(dtype=float, shape=(n,n))
        Q : The committor vector
          ndarray(dtype=float, shape=(n)
        A : set of representatives (indices defining set A in F)
          ndarray(dtype=int)
        B : set of representatives
          ndarray(dtype=int)
        """
        # F.shape[0] == F.shape[1] = n, A.shape = n = B.shape
        if F.shape[0] != F.shape[1] != Q.shape[0]:
            raise ValueError('shapes of inputs not matching')
        n = F.shape[0]
        if A.shape != B.shape != n:
            raise ValueError('shapes of input sets not matching')
        
        F = ndarray_to_stallone_array(F)
        Q = JArray(JDouble)(Q)
        A = JArray(JInt)(A)
        B = JArray(JInt)(B)
        
        # PathwayDecomposition(IDoubleArray _F, double[] _Q, int[] _A, int[] _B)
        try:
            self.PD = stallone.mc.tpt.PathwayDecomposition(F, Q, A, B)
        except Exception as e:
            log.exception('error during creation/calculation of PathwayDecomposition.', e)
            raise
    
    #public int[][] removeEdge(int[][] set, int[] edge)
    def removeEdge(self, set, edge):
        self.PD.removeEdge(set, edge)
    
    #public int[][] findGap(int[][] pathway, int[] S1, int[] S2)
    def findGap(self, pathway, S1, S2):
        self.PD.findGap(pathway, S1, S2)
    
    #public int[] edges2vertices(int[][] path)
    def edges2vertices(self, path):
        self.PD.edges2vertices(path)
    
    def computeCurrentFlux(self):
        # returns bigdecimal
        cf = self.PD.computeCurrentFlux()
        # convert java big decimal to string and pass it to python decimal
        return Decimal(str(cf))
    
    def subtractCurrentPath(self):
        self.PD.substractCurrentPath()
    
    def nextPathway(self):
        """
        Returns
        -------
        next path way : ndarray (dtype=int)
        """
        p = self.PD.nextPathway() # returns int[]
        return np.asarray(p)
    
    def getCurrentPathway(self):
        p = self.PD.getCurrentPathway() # returns int[] 
        return np.asarray(p)

    def getCurrentFlux(self):
        # returns bigdecimal
        cf = self.PD.getCurrentFlux()
        # convert java big decimal to string and pass it to python decimal
        return Decimal(str(cf))
    
    