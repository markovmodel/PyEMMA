'''
Created on 18.03.2015

@author: marscher
'''
from .transformer import Transformer
import numpy as np


class LinearOperator(Transformer):

    """Apply a linear operator to data.

    Parameters
    ----------
    A : ndarray (n, n)
        operator
    """

    def __init__(self, A):

        self.A = A.T

    def map(self, X):
        """Chunk-based application of linear operator A to X (and Y).

        Parameters
        ----------
        X : ndarray(n, m)
            the input data

        Returns
        -------
        Y : ndarray(n,)
            the projected data

        """
        return np.dot(X, self.A)

    def _parametrize(self):
        pass
