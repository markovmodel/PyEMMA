r"""Cython implementation of iterative likelihood maximization.

.. moduleauthor:: F. Paul <fabian DOT paul AT fu-berlin DOT de>

"""

import numpy
cimport numpy
import pyemma.msm.estimation
import warnings
import pyemma.util.exceptions


cdef extern from "_mle_trev_given_pi.h":
  int _mle_trev_given_pi_dense(double * const T, const double * const C, const double * const mu, const int n, const double maxerr, const int maxiter)

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
  assert pyemma.msm.estimation.is_connected(C, directed=False), 'C must be (weakly) connected'

  cdef numpy.ndarray[double, ndim=2, mode="c"] c_C = C.astype(numpy.float64, order='C', copy=False)
  cdef numpy.ndarray[double, ndim=1, mode="c"] c_mu = mu.astype(numpy.float64, order='C', copy=False)

  assert c_C.shape[0]==c_C.shape[1]==c_mu.shape[0], 'Dimensions of C and mu don\'t agree.'

  cdef numpy.ndarray[double, ndim=2, mode="c"] T = numpy.zeros_like(c_C, dtype=numpy.float64, order='C')
  
  err = _mle_trev_given_pi_dense(
        <double*> numpy.PyArray_DATA(T),
        <double*> numpy.PyArray_DATA(c_C),
        <double*> numpy.PyArray_DATA(c_mu),
        c_C.shape[0],
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

  return T

