/* moduleauthor:: F. Paul <fabian DOT paul AT fu-berlin DOT de> */
#include <stdlib.h>
#include <math.h>

#ifdef _MSC_VER
#undef isnan
int isnan(double var)
{
    volatile double d = var;
    return d != d;
}
#endif

#undef NDEBUG
#include <assert.h>
#include "_mle_trev_given_pi.h"

static double distsq(const int n, const double *const a, const double *const b)
{
  double d = 0.0;
  int i;
#pragma omp parallel for reduction(+:d)
  for(i=0; i<n; i++) {
    d += (a[i]-b[i])*(a[i]-b[i]);
  }
  return d;
}

#define C(i,j) (C [(i)*n+(j)])
#define T(i,j) (T[(i)*n+(j)])

int _mle_trev_given_pi_dense(double * const T, const long long * const C, const double * const mu, const int n, double maxerr, const int maxiter, const double eps)
{
  double d_sq;
  int i, k, l, err, iteration;
  double *lam, *lam_new, *temp;
  
  lam= (double*)malloc(n*sizeof(double));
  lam_new= (double*)malloc(n*sizeof(double));
  if(!(lam&&lam_new)) { err=1; goto error; }
  
  /* check mu */
  for(i=0; i<n; i++) {
    if(mu[i]==0) { err=4; goto error; }
  }
  
  /* check C */
  if(eps==0) {
    for(i=0; i<n; i++) {
      if(C(i,i) == 0) { err=6; goto error; }
    }
  }
  
  /* initialise lambdas */
  for(i=0; i<n; i++) {
    lam_new[i] = 0.0;
    for(k=0; k<n; k++) {
      lam_new[i] += 0.5*(C(i,k)+C(k,i));
    }
    if(lam_new[i]==0) { err=3; goto error; }
  }

  /* iterate lambdas */  
  iteration = 0;
  do {
    /* swap buffers */
    temp = lam;
    lam = lam_new;
    lam_new = temp;
    
    err = 0;

#pragma omp parallel for private(i)
    for(k=0; k<n; k++) {
      lam_new[k] = 0.0;
      for(i=0; i<n; i++) {
        double C_ik = C(i,k)+C(k,i);
        if(i==k && C_ik==0) C_ik = eps;
        lam_new[k] += C_ik / ((mu[k]*lam[i])/(mu[i]*lam[k])+1);
      }
      if(isnan(lam_new[k]) && err==0) err=2; 
    }
    
    if(err!=0) goto error;
    iteration += 1;
    d_sq = distsq(n,lam,lam_new);
  } while(d_sq > maxerr*maxerr && iteration < maxiter);
  
  if(iteration==maxiter) { err=5; goto error; } 

  /* calculate T */
  for(k=0; k<n; k++) {
     for(l=0; l<n; l++) {
        double C_kl = C(k,l)+C(l,k);
        if(k==l && C_kl==0) C_kl = eps;
        T(k,l) = C_kl / (lam_new[k] + lam_new[l]*mu[k]/mu[l]);
     }
   }

  if(lam) free(lam);
  if(lam_new) free(lam_new);
  return 0;
  
error:
  if(lam) free(lam);
  if(lam_new) free(lam_new);
  return -err;
}
