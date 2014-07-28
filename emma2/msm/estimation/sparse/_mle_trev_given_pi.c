#include <malloc.h>
#include <math.h>
#undef NDEBUG
#include <assert.h>
#include <stdio.h> /* debug */
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

int _mle_trev_given_pi(double * const T, const long long * const C, const double * const pi, const int n, double maxerr, const int maxiter, const double mu)
{
  double d_sq;
  int i, k, l, err, iteration;
  double *lam, *lam_new, *temp;
  
  lam= (double*)malloc(n*sizeof(double));
  lam_new= (double*)malloc(n*sizeof(double));
  if(!(lam&&lam_new)) { err=1; goto error; }
  
  /* check pi */
  for(i=0; i<n; i++) {
    if(pi[i]==0) { err=4; goto error; }
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

#pragma omp parallel for private(i)
    for(k=0; k<n; k++) {
      lam_new[k] = 0.0;
      for(i=0; i<n; i++) {
        // in principle should only iterate over C_ij!=0 and C_ii
        double C_ik = C(i,k)+C(k,i);
        if(i==k && C_ik==0) C_ik = mu;
        lam_new[k] += C_ik / ((pi[k]*lam[i])/(pi[i]*lam[k])+1);
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
        if(k==l && C_kl==0) C_kl = mu;
        T(k,l) = C_kl / (lam_new[k] + lam_new[l]*pi[k]/pi[l]);
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

struct coo { int i; int j; double d; };

int _mle_trev_given_pi_sparse(double * const T, const long long * const C, const double * const pi, const int n, double maxerr, const int maxiter, const double mu)
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
  
  /* check pi */
  for(i=0; i<n; i++) {
    if(pi[i]==0) { err=4; goto error; }
  }
  
  /* initialise lambdas */
  //printf("plam\n");
  for(i=0; i<n; i++) {
    lam_new[i] = 0.0;
    for(j=0; j<n; j++) {
      lam_new[i] += 0.5*(C(i,j)+C(j,i));
    }
    if(lam_new[i]==0) { err=3; goto error; }
  }
  
  /* initalize Csparse */
  //printf("count\n");
  Csparse_len = 0;
  for(i=0; i<n; i++) {
    for(j=0; j<=i; j++) {
      if(i==j || C(i,j)+C(j,i)!=0) Csparse_len++;
    }
  }
  Csparse= (struct coo*)malloc(Csparse_len*sizeof(struct coo));
  //printf("LEN=%lu * %d\n", (unsigned long)sizeof(struct coo), Csparse_len);
  if(!Csparse) { err=1; goto error; }
  t = 0;
  //printf("trans\n");
  assert(mu!=0.0);
  for(i=0; i<n; i++) {
    for(j=0; j<=i; j++) {
      if(C(i,j)+C(j,i)!=0 || i==j) {
        Csparse[t].i = i;
        Csparse[t].j = j;
        if(C(i,j)+C(j,i)==0.0) Csparse[t].d=mu; 
        else Csparse[t].d = C(i,j)+C(j,i); 
        assert(Csparse[t].d!=0);
        t++;
      }
    }
  }
  assert(t==Csparse_len);
  //printf("t=%d, Csparse_len=%d\n",t,Csparse_len);

  /* iterate lambdas */  
  iteration = 0;
  //printf("iter\n");
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
      //printf("%d/%d/%d:(%d,%d)/%d->%ld\n",t,Csparse_len,(n*n+n)/2,i,j,n,CCt_ij);
      assert(CCt_ij!=0);
      lam_new[i] += CCt_ij / ((pi[i]*lam[j])/(pi[j]*lam[i])+1.0);
      if(i!=j)
        lam_new[j] += CCt_ij / ((pi[j]*lam[i])/(pi[i]*lam[j])+1.0);
    }
    for(i=0; i<n; i++) {
       if(lam_new[i]==0) { printf("lam update -> 0\n"); err=2; goto error; }
       if(isnan(lam_new[i])) { err=2; goto error; }
    }

    iteration += 1;
    d_sq = distsq(n,lam,lam_new);
  } while(d_sq > maxerr*maxerr && iteration < maxiter);
  
  if(iteration==maxiter) { err=5; goto error; } 

  /* calculate T */
  //printf("T\n");
  for(i=0; i<n; i++) {
    for(j=0; j<n; j++) {
      T(i,j) = 0.0;
    }
  }
  for(t=0; t<Csparse_len; t++) {
      i = Csparse[t].i;
      j = Csparse[t].j;
      CCt_ij = Csparse[t].d;
      T(i,j) = CCt_ij / (lam_new[i] + lam_new[j]*pi[i]/pi[j]);
      T(j,i) = CCt_ij / (lam_new[j] + lam_new[i]*pi[j]/pi[i]);
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
