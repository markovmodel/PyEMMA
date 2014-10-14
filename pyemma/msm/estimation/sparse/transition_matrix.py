"""This module implements the transition matrix functionality"""

import numpy
import scipy.sparse


def transition_matrix_non_reversible(C):
    """implementation of transition_matrix"""
    if not scipy.sparse.issparse(C):
        C = scipy.sparse.csr_matrix(C)
    rowsum = C.tocsr().sum(axis=1)
    # catch div by zero
    if(numpy.min(rowsum) == 0.0):
        raise ValueError("matrix C contains rows with sum zero.")
    rowsum = numpy.array(1. / rowsum).flatten()
    norm = scipy.sparse.diags(rowsum, 0)
    return norm * C
