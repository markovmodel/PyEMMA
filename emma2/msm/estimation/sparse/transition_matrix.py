"""This module implements the transition matrix functionality"""

import numpy as np
import scipy.sparse


def transition_matrix(C, reversible=False, mu=None, **kwargs):
    """implementation of transition_matrix"""
    rowsum = C.tocsr().sum(axis=1)
    if(np.min(rowsum) == 0.0):
        raise ValueError("matrix C contains rows with sum zero. ")
    
    rowsum = np.array(1. / rowsum).flatten()
    norm = scipy.sparse.diags(rowsum, 0)
    return norm * C
    