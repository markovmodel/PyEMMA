r"""Cython implementation of iterative likelihood maximization.

.. moduleauthor:: F. Paul <fabian DOT paul AT fu-berlin DOT de>

"""
import numpy
import scipy
import scipy.sparse
cimport numpy
import pyemma.msm.estimation

cdef extern from "_mle_trev_given_pi.h":
    int _mle_trev_given_pi_sparse(double * const T_data,
                                  const double * const CCt_data,
                                  const long long * const i_indices,
                                  const long long * const j_indices,
                                  const int len_CCt,
                                  const double * const mu,
                                  const int len_mu,
                                  double maxerr,
                                  const int maxiter)
def check_diagonal(A):
  r"""Check matrix with non-negative entries for zero elements on diagonal.

  Parameters
  ----------
  A : (M, M) scipy.sparse matrix
      Input matrix
      
  """
  a_ii=A.diagonal()
  return numpy.all(a_ii>0.0)

def mle_trev_given_pi(
  C,
  mu,
  double maxerr = 1.0E-12,
  int maxiter = 1000000,
  double eps = 0.0
  ):

  assert maxerr>0, 'maxerr must be positive'
  assert maxiter>0, 'maxiter must be positive'
  assert eps>=0, 'eps must be non-negative'
  assert pyemma.msm.estimation.is_connected(C,directed=False), 'C must be (weakly) connected'

  CCt_csr = C+C.T
  """Convert to csr-format"""
  CCt_csr=CCt_csr.tocsr()
  """Ensure that entries are of type double"""
  CCT_csr=CCt_csr.astype(numpy.double)

  cdef numpy.ndarray[double, ndim=1, mode="c"] c_mu = mu.astype(numpy.double,order='C',copy=False)
  
  assert CCt_csr.shape[0]==CCt_csr.shape[1]==c_mu.shape[0], 'Dimensions of C and mu don\'t agree.'  
  # """add regularization"""
  # CCt_ii=CCt_csr.diagonal()
  # """Check for zero elements"""
  
  if not check_diagonal(CCt_csr) and eps==0.0:
    raise Exception('Count matrix has zero diagonal elements. Can\'t guarantee convergence of algorithm. Suggestion: set regularization parameter eps to some small value e.g. 1E-6.')
  
  """Add regularization"""
  c_ii=CCt_csr.diagonal()
  ind=(c_ii==0.0)
  prior=numpy.zeros(len(c_ii))
  prior[ind]=eps

  CCt_csr=CCt_csr+scipy.sparse.diags(prior, 0)  
  
  # for i in xrange(CCt_csr.shape[0]):
  #   if CCt_csr[i,i] == 0:
  #     if eps==0:
  #       raise Exception('Count matrix has zero diagonal elements. Can\'t guarantee convergence of algorithm. Suggestion: set regularization parameter eps to some small value e.g. 1E-6.')
  #     else:
  #       CCt_csr[i,i] = eps

  # convert to coo format 
  CCt_coo = CCt_csr.tocoo()
  n_data = CCt_coo.nnz
  cdef numpy.ndarray[double, ndim=1, mode="c"] CCt_data =  CCt_coo.data.astype(numpy.double,order='C',copy=False)
  cdef numpy.ndarray[long long, ndim=1, mode="c"] i_indices = CCt_coo.row.astype(numpy.int64,order='C',copy=True)
  cdef numpy.ndarray[long long, ndim=1, mode="c"] j_indices = CCt_coo.col.astype(numpy.int64,order='C',copy=True)
  
  # prepare data array of T in coo format
  cdef numpy.ndarray[double, ndim=1, mode="c"] T_data = numpy.zeros(n_data,dtype=numpy.double,order='C')
  
  err = _mle_trev_given_pi_sparse(
        <double*> numpy.PyArray_DATA(T_data),
        <double*> numpy.PyArray_DATA(CCt_data),
        <long long*> numpy.PyArray_DATA(i_indices),
        <long long*> numpy.PyArray_DATA(j_indices),
        n_data,
        <double*> numpy.PyArray_DATA(c_mu),
        CCt_csr.shape[0],
        maxerr,
        maxiter)
        
  # TODO: add self test: check if stationary distribution is ok
  
  if err==-1:
    raise Exception('Out of memory.')
  elif err==-2:
    raise Exception('The update of the Lagrange multipliers produced zero or NaN.')
  elif err==-3:
    raise Exception('Some row and corresponding column of C have zero counts.')
  elif err==-4:
    raise Exception('Some element of pi is zero.')
  elif err==-5:
    raise Exception('Didn\'t converge.')
  elif err==-6:
    raise Exception('Count matrix has zero diagonal elements. Can\'t guarantee convergence of algorithm. Suggestion: set regularization parameter eps to some small value e.g. 1E-6.')

  # T matrix has the same shape and positions of nonzero elements as the regularized C matrix
  return scipy.sparse.csr_matrix((T_data,(i_indices,j_indices)),shape=CCt_csr.shape)
