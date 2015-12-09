#include "_tram_direct.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

#define LAGRANGIAN_MULT_LOWER_BOUND 1.E-100

/* direct space implementation */

void _update_lagrangian_mult(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int* state_counts,
    int n_therm_states, int n_conf_states, int iteration, double *new_lagrangian_mult)
{
    int i, j, K, KMM, Ki, Kj;
    int CCT_Kij;

    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K*n_conf_states*n_conf_states;
        for(i=0; i<n_conf_states; ++i)
        {
            Ki = K*n_conf_states + i;
            new_lagrangian_mult[Ki] = 0;
            if(0 == state_counts[Ki]) continue;
            for(j=0; j<n_conf_states; ++j)
            {
                CCT_Kij = count_matrices[KMM + i*n_conf_states + j];
                if(i == j) {
                    if(lagrangian_mult[Ki]<CCT_Kij) fprintf(stderr, "Not a valid nu iterate at K=%d, i=%d.\n", K, i);
                    new_lagrangian_mult[Ki] += CCT_Kij;
                } else {
                    Kj = K*n_conf_states + j;
                    CCT_Kij += count_matrices[KMM + j*n_conf_states + i];
                    if(0 < CCT_Kij) {
                        /* one of the nus can be zero */
                        if(lagrangian_mult[Ki]+lagrangian_mult[Kj] < CCT_Kij) fprintf(stderr, "Not a valid nu iterate at K=%d, i=%d, j=%d in iteration %d.\n", K,i,j,iteration);
                        new_lagrangian_mult[Ki] += 
                            (double)CCT_Kij / (1.0 + 
                                               (lagrangian_mult[Kj]/lagrangian_mult[Ki])*
                                               (biased_conf_weights[Ki]/biased_conf_weights[Kj]));
                        if(new_lagrangian_mult[Ki]< LAGRANGIAN_MULT_LOWER_BOUND) {
                            new_lagrangian_mult[Ki] = LAGRANGIAN_MULT_LOWER_BOUND;
                        }
                    }
                }
            }
        }
    }
}

void _update_biased_conf_weights(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, double *bias_sequence,
    int *state_sequence, int *state_counts, int seq_length, double *R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_TM, double *new_biased_conf_weights)
{
    int i, j, K, Ki, Kj, KMM, x;
    int CCT_Kij, CKi;
    double divisor;

    /* compute R */
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K*n_conf_states*n_conf_states;
        for(i=0; i<n_conf_states; ++i)
        {
            CKi = 0;
            Ki = K*n_conf_states + i;
            R_K_i[Ki] = 0.0;
            if(0 == state_counts[Ki]) continue;
            for(j=0; j<n_conf_states; ++j)
            {
                CCT_Kij = count_matrices[KMM + i*n_conf_states + j];
                CKi += count_matrices[KMM + j*n_conf_states + i];
                Kj = K*n_conf_states + j;
                if(i == j) {
                    R_K_i[Ki] += CCT_Kij;
                } else {
                    CCT_Kij += count_matrices[KMM + j*n_conf_states + i];
                    if(0 < CCT_Kij) {
                        /* one of the nus can be zero */
                        if(lagrangian_mult[Ki]==0 && lagrangian_mult[Kj]==0) fprintf(stderr, "R:Warning nu[%d,%d]=nu[%d,%d]=0\n",K,i,K,j);
                        if(biased_conf_weights[Ki]==0) fprintf(stderr, "R:Warning Z[%d,%d]=0\n",K,i);
                        if(biased_conf_weights[Kj]==0) fprintf(stderr, "R:Warning Z[%d,%d]=0\n",K,j);
                        R_K_i[Ki] += (double)CCT_Kij / (1.0 +
                                                     (lagrangian_mult[Ki]/lagrangian_mult[Kj])*
                                                     (biased_conf_weights[Kj]/biased_conf_weights[Ki]));
                    }
                }
            }
            R_K_i[Ki] += state_counts[Ki] - CKi;
            if(0 < R_K_i[Ki]) R_K_i[Ki] /= biased_conf_weights[Ki];
        }
    }

    /* actual update */
    for(i=0; i < n_conf_states*n_therm_states; ++i) {
        new_biased_conf_weights[i] = 0.0;
    }

    for(x=0; x < seq_length; ++x) 
    {
        i = state_sequence[x];
        divisor = 0;
        /* calulate normal divisor */
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            if(0 < R_K_i[Ki]) {
                divisor += R_K_i[Ki]*bias_sequence[K*seq_length + x];
            }
        }
        if(divisor==0) fprintf(stderr, "divisor is zero. should never happen!\n");
        if(isnan(divisor)) fprintf(stderr, "divisor is NaN. should never happen!\n");

        /* update normal weights */
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            new_biased_conf_weights[Ki] += bias_sequence[K*seq_length + x]/divisor;
            if(isnan(new_biased_conf_weights[Ki])) fprintf(stderr, "Z:Warning Z[%d,%d]=NaN (%f,%f) %d\n",K,i,bias_sequence[K*seq_length + x],divisor,x);
            if(isinf(new_biased_conf_weights[Ki])) fprintf(stderr, "Z:Warning Z[%d,%d]=Inf (%f,%f) %d\n",K,i,bias_sequence[K*seq_length + x],divisor,x);
        }
    }
}

void _dtram_like_update(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int *state_counts, 
    int n_therm_states, int n_conf_states, double *scratch_M, int *scratch_M_int, double *new_biased_conf_weights)
{
    int K, KMM, i, j, CCT_Kji, Ki, Kj;
    double *divisor;
    int *Csum;

    divisor = scratch_M;
    Csum = scratch_M_int;
    for(i=0; i<n_conf_states; ++i)
    {
        divisor[i] = 0;
        Csum[i] = 0;
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            KMM = K*n_conf_states*n_conf_states;
            for(j=0; j<n_conf_states; ++j)
            {
                Kj = K*n_conf_states + j;
                CCT_Kji = count_matrices[KMM + j*n_conf_states + i];
                Csum[i] += CCT_Kji;
                if(i == j) {
                    divisor[i] += CCT_Kji;
                } else {
                    CCT_Kji += count_matrices[KMM + i*n_conf_states + j];
                    if(0 < CCT_Kji) {
                        divisor[i] += (double)CCT_Kji / (1.0 +
                                                        (lagrangian_mult[Ki]/lagrangian_mult[Kj])*
                                                        (biased_conf_weights[Kj]/biased_conf_weights[Ki]));
                    }
                }
            }
        }
    }

    for(i=0; i<n_conf_states; ++i)
    {
        if(0 < Csum[i]) {
            for(K=0; K<n_therm_states; ++K)
            {
                    Ki = K*n_conf_states + i;
                    new_biased_conf_weights[Ki] = biased_conf_weights[Ki] * Csum[i] / divisor[i];
            }
        }
    }
}
