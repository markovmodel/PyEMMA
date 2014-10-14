'''
Created on 18.10.2013

@author: marscher
'''
import numpy as np
from decimal import Decimal # for wrapping java.math.BigDecimal

from pyemma.util.log import getLogger
from pyemma.util.pystallone import ndarray_to_stallone_array, stallone, JArray, JInt, JDouble

__all__ = ['PathwayDecomposition']

log = getLogger()


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
        if np.shape(F)[0] != np.shape(F)[1] != np.shape(Q)[0]:
            raise ValueError('shapes of inputs not matching')

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
        if p is None:
            return None
        else:
            return np.asarray(p)

    def getCurrentPathway(self):
        p = self.PD.getCurrentPathway() # returns int[] 
        return np.asarray(p)

    def getCurrentFlux(self):
        # returns bigdecimal
        cf = self.PD.getCurrentFlux()
        # convert java big decimal to string and pass it to python decimal
        return Decimal(str(cf))
