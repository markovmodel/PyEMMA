r"""Cython implementation of iterative likelihood maximization.

.. moduleauthor:: F. Paul <fabian DOT paul AT fu-berlin DOT de>

"""

import numpy
import scipy
import scipy.sparse
cimport numpy
import pyemma.msm.estimation
import warnings
import pyemma.util.exceptions

cdef extern from "_mle_trev.h":
  int _mle_trev_sparse(double * const T_data, const double * const CCt_data, const int * const i_indices, const int * const j_indices, const int len_CCt, const double * const sum_C, const int dim, const double maxerr, const int maxiter)

def mle_trev(C, double maxerr = 1.0E-12, int maxiter = 1000000):

  assert maxerr > 0, 'maxerr must be positive'
  assert maxiter > 0, 'maxiter must be positive'
  assert C.shape[0] == C.shape[1], 'C must be a square matrix.'
  assert pyemma.msm.estimation.is_connected(C, directed=True), 'C must be strongly connected'
  
  C_sum_py = C.sum(axis=1).A1
  cdef numpy.ndarray[double, ndim=1, mode="c"] C_sum = C_sum_py.astype(numpy.float64, order='C', copy=False)

  CCt = C+C.T
  # convert CCt to coo format 
  CCt_coo = CCt.tocoo()
  n_data = CCt_coo.nnz
  cdef numpy.ndarray[double, ndim=1, mode="c"] CCt_data =  CCt_coo.data.astype(numpy.float64, order='C', copy=False)
  cdef numpy.ndarray[int, ndim=1, mode="c"] i_indices = CCt_coo.row.astype(numpy.intc, order='C', copy=True)
  cdef numpy.ndarray[int, ndim=1, mode="c"] j_indices = CCt_coo.col.astype(numpy.intc, order='C', copy=True)
  
  # prepare data array of T in coo format
  cdef numpy.ndarray[double, ndim=1, mode="c"] T_data = numpy.zeros(n_data, dtype=numpy.float64, order='C')
  
  err = _mle_trev_sparse(
        <double*> numpy.PyArray_DATA(T_data),
        <double*> numpy.PyArray_DATA(CCt_data),
        <int*> numpy.PyArray_DATA(i_indices),
        <int*> numpy.PyArray_DATA(j_indices),
        n_data,
        <double*> numpy.PyArray_DATA(C_sum),
        CCt.shape[0],
        maxerr,
        maxiter)
  
  if err == -1:
    raise Exception('Out of memory.')
  elif err == -2:
    raise Exception('The update of the stationary distribution produced zero or NaN.')
  elif err == -3:
    raise Exception('Some row and corresponding column of C have zero counts.')
  elif err == -5:
    warnings.warn('Reversible transition matrix estimation didn\'t converge.', pyemma.util.exceptions.NotConvergedWarning)

  # T matrix has the same shape and positions of nonzero elements as CCt
  return scipy.sparse.csr_matrix((T_data,(i_indices,j_indices)),shape=CCt.shape)
