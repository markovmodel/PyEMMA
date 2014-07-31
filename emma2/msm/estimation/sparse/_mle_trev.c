#include <malloc.h>
#include <math.h>
#undef NDEBUG
#include <assert.h>
#include "_mle_trev.h"

static double distsq(const int n, const double *const a, const double *const b)
{
  double d = 0.0;
  int i;
  for(i=0; i<n; i++) {
    d += (a[i]-b[i])*(a[i]-b[i]);
  }
  return d;
}

int _mle_trev_sparse(double * const T_data, const long long * const CCt_data, const long long * const i_indices, const long long * const j_indices, const int len_CCt, const long long * const sum_C, const int dim, const double maxerr, const int maxiter)
{
  double d_sq;
  int i, j, t, err, iteration;
  double *x, *x_new, *x_sum, *temp;
  double CCt_ij, x_norm;

  err = 0;

  x = (double*)malloc(len_CCt*sizeof(double));
  x_new= (double*)malloc(len_CCt*sizeof(double));
  x_sum= (double*)malloc(dim*sizeof(double));
  if(!(x&&x_new&&x_sum)) { err=1; goto error; }
  
  /* initialize x */
  for(t = 0; t<len_CCt; t++) x_new[t] = 0.5*CCt_data[t];

  /* iterate */
  iteration = 0;
  do {
    /* swap buffers */
    temp = x;
    x = x_new;
    x_new = temp;
    
    /* update x_sum */
    for(i = 0; i<dim; i++) x_sum[i] = 0;
    for(t = 0; t<len_CCt; t++) { 
      i = i_indices[t];
      j = j_indices[t];
      x_sum[i] += x[t];
      if(i!=j) x_sum[j] += x[t];
    }  

    /* update x */
    x_norm = 0;
    for(t=0; t<len_CCt; t++) {
      i = i_indices[t];
      j = j_indices[t];
      CCt_ij = CCt_data[t];
      assert(CCt_ij!=0);
      x[t] = CCt_ij / ( sum_C[i]/x_sum[i] + sum_C[j]/x_sum[j] );
      x_norm += x[t];
    }
    
    /* normalize x */
    for(t=0; t<len_CCt; t++) {
      x[t]/=x_norm;
    }

    iteration += 1;
    d_sq = distsq(len_CCt,x,x_new);
  } while(d_sq > maxerr*maxerr && iteration < maxiter);
  
  if(iteration==maxiter) { err=5; goto error; } 

  /* calculate T */
  for(t=0; t<len_CCt; t++) {
      i = i_indices[t];
      j = j_indices[t];
      T_data[t] = x[t] / x_sum[i];
  }

  free(x);
  free(x_new);
  free(x_sum);
  return 0;
  
error:
  free(x);
  free(x_new);
  free(x_sum);
  return -err;
}
