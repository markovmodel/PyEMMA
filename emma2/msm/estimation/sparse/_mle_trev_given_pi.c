#include <malloc.h>
#include <math.h>
#undef NDEBUG
#include <assert.h>
#include "_mle_trev_given_pi.h"

static double distsq(const int n, const double *const a, const double *const b)
{
  double d = 0.0;
  int i;
  for(i=0; i<n; i++) {
    d += (a[i]-b[i])*(a[i]-b[i]);
  }
  return d;
}

#define C(i,j) (C [(i)*n+(j)])
#define T(i,j) (T[(i)*n+(j)])

struct coo { int i; int j; double d; };

int _mle_trev_given_pi_sparse(double * const T, const long long * const C, const double * const mu, const int n, double maxerr, const int maxiter, const double eps)
{
  double d_sq;
  int i, j, t, err, iteration;
  int Csparse_len;
  double *lam, *lam_new, *temp;
  double CCt_ij;
  struct coo *Csparse;

  Csparse = NULL;
  err = 0;

  lam= (double*)malloc(n*sizeof(double));
  lam_new= (double*)malloc(n*sizeof(double));
  if(!(lam&&lam_new)) { err=1; goto error; }

  /* check mu */
  for(i=0; i<n; i++) {
    if(mu[i]==0) { err=4; goto error; }
  }

  /* initialise lambdas */
  for(i=0; i<n; i++) {
    lam_new[i] = 0.0;
    for(j=0; j<n; j++) {
      lam_new[i] += 0.5*(C(i,j)+C(j,i));
    }
    if(lam_new[i]==0) { err=3; goto error; }
  }
  
  /* initalize Csparse */
  Csparse_len = 0;
  for(i=0; i<n; i++) {
    for(j=0; j<=i; j++) {
      if(i==j || C(i,j)+C(j,i)!=0) Csparse_len++;
    }
  }
  Csparse= (struct coo*)malloc(Csparse_len*sizeof(struct coo));
  if(!Csparse) { err=1; goto error; }
  t = 0;
  assert(eps!=0.0);
  for(i=0; i<n; i++) {
    for(j=0; j<=i; j++) {
      if(C(i,j)+C(j,i)!=0 || i==j) {
        Csparse[t].i = i;
        Csparse[t].j = j;
        if(C(i,j)+C(j,i)==0.0) Csparse[t].d=eps; 
        else Csparse[t].d = C(i,j)+C(j,i); 
        assert(Csparse[t].d!=0);
        t++;
      }
    }
  }
  assert(t==Csparse_len);

  /* iterate lambdas */  
  iteration = 0;
  do {
    /* swap buffers */
    temp = lam;
    lam = lam_new;
    lam_new = temp;

    for(i=0; i<n; i++) {
       lam_new[i] = 0.0;
    }
    for(t=0; t<Csparse_len; t++) {
      i = Csparse[t].i;
      j = Csparse[t].j;
      CCt_ij = Csparse[t].d;
      assert(CCt_ij!=0);
      lam_new[i] += CCt_ij / ((mu[i]*lam[j])/(mu[j]*lam[i])+1.0);
      if(i!=j)
        lam_new[j] += CCt_ij / ((mu[j]*lam[i])/(mu[i]*lam[j])+1.0);
    }
    for(i=0; i<n; i++) {
       if(lam_new[i]==0) { err=2; goto error; }
       if(isnan(lam_new[i])) { err=2; goto error; }
    }

    iteration += 1;
    d_sq = distsq(n,lam,lam_new);
  } while(d_sq > maxerr*maxerr && iteration < maxiter);
  
  if(iteration==maxiter) { err=5; goto error; } 

  /* calculate T */
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      T(i,j) = 0.0;
    }
  }
  for(t=0; t<Csparse_len; t++) {
      i = Csparse[t].i;
      j = Csparse[t].j;
      CCt_ij = Csparse[t].d;
      T(i,j) = CCt_ij / (lam_new[i] + lam_new[j]*mu[i]/mu[j]);
      T(j,i) = CCt_ij / (lam_new[j] + lam_new[i]*mu[j]/mu[i]);
  }

  if(lam) free(lam);
  if(lam_new) free(lam_new);
  if(Csparse) free(Csparse);
  return 0;
  
error:
  if(lam) free(lam);
  if(lam_new) free(lam_new);
  if(Csparse) free(Csparse);
  return -err;
}
