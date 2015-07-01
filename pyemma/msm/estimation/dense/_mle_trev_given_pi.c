/* * Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
 * Berlin, 14195 Berlin, Germany.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

int _mle_trev_given_pi_dense(double * const T, const double * const C, const double * const mu, const int n, const double maxerr, const int maxiter)
{
  double d_sq, norm, C_ij;
  int i, j, err, iteration;
  double *lam, *lam_new, *temp;
  
  lam= (double*)malloc(n*sizeof(double));
  lam_new= (double*)malloc(n*sizeof(double));
  if(!(lam && lam_new)) { err=1; goto error; }

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

  /* iterate lambdas */  
  iteration = 0;
  do {
    /* swap buffers */
    temp = lam;
    lam = lam_new;
    lam_new = temp;
    
    err = 0;

#pragma omp parallel for private(i,C_ij)
    for(j=0; j<n; j++) {
      lam_new[j] = 0.0;
      for(i=0; i<n; i++) {
        C_ij = C(i,j)+C(j,i);
        if(C_ij==0) continue;
        lam_new[j] += C_ij / ((mu[j]*lam[i])/(mu[i]*lam[j])+1);
      }
      if(isnan(lam_new[j]) && err==0) err=2; 
    }
    
    if(err!=0) goto error;
    iteration += 1;
    d_sq = distsq(n,lam,lam_new);
  } while(d_sq > maxerr*maxerr && iteration < maxiter);
  
  if(iteration==maxiter) { err=5; goto error; } 

  /* calculate T */
  for(i=0; i<n; i++) {
    norm = 0;
    for(j=0; j<n; j++) {
      C_ij = C(i,j)+C(j,i);
      if(i!=j) {
        T(i,j) = C_ij / (lam_new[i] + lam_new[j]*mu[i]/mu[j]);
        norm += T(i,j);
      }
    }
    if(norm>1.0) T(i,i) = 0.0; else T(i,i) = 1.0-norm;
  }

  if(lam) free(lam);
  if(lam_new) free(lam_new);
  return 0;
  
error:
  if(lam) free(lam);
  if(lam_new) free(lam_new);
  return -err;
}
