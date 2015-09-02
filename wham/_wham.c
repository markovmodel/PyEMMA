/*
*   Copyright 2015 Christoph Wehmeyer
*/

#include <math.h>
#include "../lse/_lse.h"
#include "_wham.h"

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

extern void _wham_fi(
    double *log_N_K, double *log_N_i, double *f_K, double *b_K_i,
    int n_therm_states, int n_markov_states, double *scratch_T, double *f_i)
{
    int i, K;
    for(i=0; i<n_markov_states; ++i)
    {
        for(K=0; K<n_therm_states; ++K)
            scratch_T[K] = log_N_K[K] - b_K_i[K*n_markov_states + i] + f_K[K];
        f_i[i] = _logsumexp(scratch_T, n_therm_states) - log_N_i[i];
    }
}

extern void _wham_fk(
    double *f_i, double *b_K_i, int n_therm_states, int n_markov_states,
    double *scratch_M, double *f_K)
{
    int i, K, KM;
    for(K=0; K<n_therm_states; ++K)
    {
        KM = K*n_markov_states;
        for(i=0; i<n_markov_states; ++i)
            scratch_M[i] = -(b_K_i[KM + i] + f_i[i]);
        f_K[K] = -_logsumexp(scratch_M, n_markov_states);
    }
}

extern void _wham_normalize(double *f_i, int n_markov_states, double *scratch_M)
{
    int i;
    double shift;
    for(i=0; i<n_markov_states; ++i)
        scratch_M[i] = -f_i[i];
    shift = _logsumexp(scratch_M, n_markov_states);
    for(i=0; i<n_markov_states; ++i)
        f_i[i] += shift;
}

