'''
Created on 24.09.2013

@author: marscher
'''
import numpy as np


def isProbabilisticMatrix(T):
    """
    returns whether T is a probabilistic matrix
    """
    if isinstance(T, np.ndarray):
        return (T >= 0.0).any() and (T <= 1).any()
    else:
        raise NotImplemented("only impled for NumPy ndarray type")

def isTransitionMatrix(T):
    """
    returns whether T is a transition matrix
    """
    return isProbabilisticMatrix(T)
