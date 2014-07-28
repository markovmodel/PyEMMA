import numpy
cimport numpy

cdef extern from "_mle_trev_given_pi.h":
  int _mle_trev_given_pi(double * const T, const long long * const C, const double * const pi, const int n, double maxerr, const int maxiter, const double mu)
  int _mle_trev_given_pi_sparse(double * const T, const long long * const C, const double * const pi, const int n, double maxerr, const int maxiter, const double mu)


def mle_trev_given_pi(
  C,
  pi,
  double maxerr = 1.0E-12,
  int maxiter = 1000000,
  double mu = 1.0E-6,
  bint sparse = True
  ):

  assert maxerr>0
  assert maxiter>0
  assert mu>0

  #cdef numpy.ndarray[int, ndim=2, mode="c"] Cint = C.astype(numpy.int,order='C',copy=False)

  cdef numpy.ndarray[long long, ndim=2, mode="c"] Cint = C.astype(numpy.int64,order='C',copy=False)
  cdef numpy.ndarray[double, ndim=1, mode="c"] Ppi = pi.astype(numpy.double,order='C',copy=False)

  assert Cint.shape[0]==Cint.shape[1]==Ppi.shape[0]

  cdef numpy.ndarray[double, ndim=2, mode="c"] T = numpy.zeros_like(Cint,dtype=numpy.double,order='C')
  
  if sparse:
    err = _mle_trev_given_pi_sparse(
        <double*> numpy.PyArray_DATA(T),
        <long long*> numpy.PyArray_DATA(Cint),
        <double*> numpy.PyArray_DATA(Ppi),
        C.shape[0],
        maxerr,
        maxiter,
        mu)
  else:
    err = _mle_trev_given_pi(
        <double*> numpy.PyArray_DATA(T),
        <long long*> numpy.PyArray_DATA(Cint),
        <double*> numpy.PyArray_DATA(Ppi),
        C.shape[0],
        maxerr,
        maxiter,
        mu)
        
  # TODO: add self test: check if stationary distribution is ok
  
  
  if err==-1:
    raise Exception('Out of memeory.')
  elif err==-2:
    raise Exception('The update of the Lagrange multipliers produced NaN.')
  elif err==-3:
    raise Exception('Some row and corresponding column of C have zero counts.')
  elif err==-4:
    raise Exception('Some element of pi is zero.')
  elif err==-5:
    raise Exception('Didn\'t converge.')
    
     
  return T

