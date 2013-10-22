
from numpy import allclose
from scipy.sparse.csr import isspmatrix_csr

def allclose_sparse(A, B, rtol=1e-5, atol=1e-9):
    """
    Compares two sparse matrices in the same matter like numpy.allclose()
    Parameters
    ----------
    A : scipy.sparse matrix
        first matrix to compare
    B : scipy.sparse matrix
        second matrix to compare
    rtol : float
        relative tolerance
    atol : float
        absolute tolerance
    
    Returns
    -------
    True, if given matrices are equal in bounds of rtol and atol
    False, otherwise
    """
    if not isspmatrix_csr(A):
        A = A.tocsr()
    if not isspmatrix_csr(B):
        B = B.tocsr()
    
    close_values = allclose(A.data, B.data, rtol=rtol, atol=atol)
    equal_inds = (A.indices == B.indices).all()
    equal_indptr = (A.indptr == B.indptr).all()
    
    return close_values and equal_inds and equal_indptr
