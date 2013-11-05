"""This module implements the transition matrix functionality"""

import numpy as np
import scipy.sparse

def transition_matrix_non_reversible(C):
    """implementation of transition_matrix"""
    if scipy.sparse.issparse(C):
        C=C
    else:
        C=scipy.sparse.csr_matrix(C)    
    rowsum = C.tocsr().sum(axis=1)
    if(np.min(rowsum) == 0.0):
        raise ValueError("matrix C contains rows with sum zero. ")    
    rowsum = np.array(1. / rowsum).flatten()
    norm = scipy.sparse.diags(rowsum, 0)
    return norm * C
    
