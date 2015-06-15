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

cdef extern from "_mle_trev_given_pi.h":
    int _mle_trev_given_pi_sparse(double * const T_unnormalized_data,
                                  const double * const CCt_data,
                                  const int * const i_indices,
                                  const int * const j_indices,
                                  const int len_CCt,
                                  const double * const mu,
                                  const int len_mu,
                                  const double maxerr,
                                  const int maxiter)

def mle_trev_given_pi(
  C,
  mu,
  double maxerr = 1.0E-12,
  int maxiter = 1000000,
  double eps = 0.0
  ):

  assert maxerr > 0, 'maxerr must be positive'
  assert maxiter > 0, 'maxiter must be positive'
  assert eps >= 0, 'eps must be non-negative'
  if eps>0:
     warnings.warn('A regularization parameter value eps!=0 is not necessary for convergence. The parameter will be removed in future versions.', DeprecationWarning)
  assert pyemma.msm.estimation.is_connected(C,directed=False), 'C must be (weakly) connected'

  cdef numpy.ndarray[double, ndim=1, mode="c"] c_mu = mu.astype(numpy.float64, order='C', copy=False)

  CCt_coo = (C+C.T).tocoo()

  assert CCt_coo.shape[0] == CCt_coo.shape[1] == c_mu.shape[0], 'Dimensions of C and mu don\'t agree.'

  n_data = CCt_coo.nnz
  cdef numpy.ndarray[double, ndim=1, mode="c"] CCt_data =  CCt_coo.data.astype(numpy.float64, order='C', copy=False)
  cdef numpy.ndarray[int, ndim=1, mode="c"] i_indices = CCt_coo.row.astype(numpy.intc, order='C', copy=False)
  cdef numpy.ndarray[int, ndim=1, mode="c"] j_indices = CCt_coo.col.astype(numpy.intc, order='C', copy=False)

  # prepare data array of T in coo format
  cdef numpy.ndarray[double, ndim=1, mode="c"] T_unnormalized_data = numpy.zeros(n_data, dtype=numpy.float64, order='C')

  err = _mle_trev_given_pi_sparse(
        <double*> numpy.PyArray_DATA(T_unnormalized_data),
        <double*> numpy.PyArray_DATA(CCt_data),
        <int*> numpy.PyArray_DATA(i_indices),
        <int*> numpy.PyArray_DATA(j_indices),
        n_data,
        <double*> numpy.PyArray_DATA(c_mu),
        CCt_coo.shape[0],
        maxerr,
        maxiter)

  if err == -1:
    raise Exception('Out of memory.')
  elif err == -2:
    raise Exception('The update of the Lagrange multipliers produced NaN.')
  elif err == -3:
    raise Exception('Some row and corresponding column of C have zero counts.')
  elif err == -4:
    raise Exception('Some element of pi is zero.')
  elif err == -5:
    warnings.warn('Reversible transition matrix estimation with fixed stationary distribution didn\'t converge.', pyemma.util.exceptions.NotConvergedWarning)

  # unnormalized T matrix has the same shape and positions of nonzero elements as the C matrix
  T_unnormalized = scipy.sparse.csr_matrix((T_unnormalized_data, (i_indices.copy(), j_indices.copy())), shape=CCt_coo.shape)
  # finish T by setting the diagonal elements according to the normalization constraint
  rowsum = T_unnormalized.sum(axis=1).A1
  T_diagonal = scipy.sparse.diags(numpy.maximum(1.0-rowsum,0.0), 0)
  
  return T_unnormalized + T_diagonal
