#include "_tram_direct.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

#define THERMOTOOLS_TRAM_PRIOR 0
//1.0E-10
#define THRESH 1.E-5
#define SMALL 1.E-8

/* direct space implementation */

void _update_lagrangian_mult(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int* state_counts,
    int n_therm_states, int n_conf_states, double *new_lagrangian_mult)
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
            // if(lagrangian_mult[Ki]>0 && lagrangian_mult[Ki]<1 && state_counts[Ki]>0) fprintf(stderr, "Example 0<nu[%d,%d]<1: %g\n", K,i,lagrangian_mult[Ki]);
            if(0 == state_counts[Ki]) continue;
            for(j=0; j<n_conf_states; ++j)
            {
                CCT_Kij = count_matrices[KMM + i*n_conf_states + j];
                if(i == j) {
                    if(lagrangian_mult[Ki]<CCT_Kij) fprintf(stderr, "Not a valid nu iterate at K=%d, i=%d", K, i);
                    new_lagrangian_mult[Ki] += CCT_Kij + THERMOTOOLS_TRAM_PRIOR;
                } else {
                    Kj = K*n_conf_states + j;
                    CCT_Kij += count_matrices[KMM + j*n_conf_states + i];
                    if(0 < CCT_Kij) {
                        /* one of the nus can be zero */
                        if(lagrangian_mult[Ki]==0) fprintf(stderr, "Warning nu[%d,%d]=0\n",K,i);
                        if(lagrangian_mult[Kj]==0) fprintf(stderr, "Warning nu[%d,%d]=0\n",K,j);
                        if(lagrangian_mult[Ki]==0 && lagrangian_mult[Kj]==0) fprintf(stderr, "Warning nu[%d,%d]=nu[%d,%d]=0\n",K,i,K,j);
                        if(lagrangian_mult[Ki]+lagrangian_mult[Kj]<CCT_Kij) fprintf(stderr, "Not a valid nu iterate at K=%d, i=%d, j=%d.\n", K,i,j);
                        if(biased_conf_weights[Ki]==0) fprintf(stderr, "Warning Z[%d,%d]=0\n",K,i);
                        if(biased_conf_weights[Kj]==0) fprintf(stderr, "Warning Z[%d,%d]=0\n",K,j);
                        new_lagrangian_mult[Ki] += 
                            (double)CCT_Kij / (1.0 + 
                                               (lagrangian_mult[Kj]/lagrangian_mult[Ki])*
                                               (biased_conf_weights[Ki]/biased_conf_weights[Kj]));
                    }
                }
                    
            }
        }
    }

    /* apply fixed */
    for(K=0; K<n_therm_states; ++K) {
        for(i=0; i<n_conf_states; ++i) {
            Ki = K*n_conf_states + i;
            if(state_counts[Ki] > 0) {
                for(j=0; j<n_conf_states; j++) {
                    Kj = K*n_conf_states + j;
                    CCT_Kij = count_matrices[K*n_conf_states*n_conf_states + i*n_conf_states + j] +
                              count_matrices[K*n_conf_states*n_conf_states + j*n_conf_states + i];
                    if(new_lagrangian_mult[Ki]+new_lagrangian_mult[Kj] < CCT_Kij) {
                        if(new_lagrangian_mult[Ki]<new_lagrangian_mult[Kj]) {
                             /* add correction to the smaller one */
                             fprintf(stderr, "Fixing nu[%d,%d] to match C[%d,%d,%d].\n", K,i,K,i,j);
                             /* new_lagrangian_mult[Ki] = CCT_Kij - new_lagrangian_mult[Kj] + 1.E-10; */
                        } /* else do nothing */
                    }
                }
            }
        }
    }

}

struct my_sparse _update_biased_conf_weights(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, double *bias_sequence,
    int *state_sequence, int *state_counts, int seq_length, double *R_K_i,
    int n_therm_states, int n_conf_states, int check_overlap, double *new_biased_conf_weights)
{
    int i, j, K, Ki, Kj, KMM, x, n, L, Li, KLi;
    int CCT_Kij, CKi;
    double divisor, sum;
    double *skipped_biased_conf_weights, *skipped_divisor;
    int *rows, *cols;
    struct my_sparse s;
    
    s.length = 0;

    /* compute R */
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K*n_conf_states*n_conf_states;
        for(i=0; i<n_conf_states; ++i)
        {
            CKi = 0;
            Ki = K*n_conf_states + i;
            R_K_i[Ki] = 0.0;
            for(j=0; j<n_conf_states; ++j)
            {
                CCT_Kij = count_matrices[KMM + i*n_conf_states + j];
                CKi += count_matrices[KMM + j*n_conf_states + i];
                Kj = K*n_conf_states + j;
                if(i == j) {
                    R_K_i[Ki] += CCT_Kij + THERMOTOOLS_TRAM_PRIOR;
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
        }
    }

    if(check_overlap) {
        /* initialize skipped_biased_conf_weights */
        skipped_biased_conf_weights = malloc(n_therm_states*n_therm_states*n_conf_states*sizeof(double));
        assert(skipped_biased_conf_weights);
        for(i=0; i < n_therm_states*n_therm_states*n_conf_states; ++i) {
            skipped_biased_conf_weights[i] = 0.0;
        }
        skipped_divisor =  malloc(n_therm_states*sizeof(double));
        assert(skipped_divisor);
    }

    /* actual update */
    for(i=0; i < n_conf_states*n_therm_states; ++i) new_biased_conf_weights[i] = 0.0;
    for(x=0; x < seq_length; ++x) 
    {
        i = state_sequence[x];
        assert(i<n_conf_states);
        divisor = 0;
        /* calulate normal divisor */
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            if(THERMOTOOLS_TRAM_PRIOR < R_K_i[Ki]) divisor += R_K_i[Ki]*bias_sequence[K*seq_length + x]/biased_conf_weights[Ki];
        }
        if(divisor==0) fprintf(stderr, "divisor is zero. should never happen!\n");
        if(isnan(divisor)) fprintf(stderr, "divisor is NaN. should never happen!\n");

        if(check_overlap) {
            /* calculate skipped divisors */
            for(L=0; L<n_therm_states; ++L) {
                Li = L*n_conf_states + i;
                if(R_K_i[Li]>SMALL) {
                    skipped_divisor[L] = 0;
                    for(K=0; K<n_therm_states; ++K) 
                    {
                        Ki = K*n_conf_states + i;
                        if(K!=L && R_K_i[Ki]>0) skipped_divisor[L] += R_K_i[Ki]*bias_sequence[K*seq_length + x]/biased_conf_weights[Ki];
                    }
                }
            }
        }

        /* update normal weights */
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            new_biased_conf_weights[Ki] += bias_sequence[K*seq_length + x]/divisor;
            if(isnan(new_biased_conf_weights[Ki])) { fprintf(stderr, "Z:Warning Z[%d,%d]=NaN (%f,%f) %d\n",K,i,bias_sequence[K*seq_length + x],divisor,x); exit(1); }
        }

        if(check_overlap) {
            /* update skipped weights */
            for(L=0; L<n_therm_states; ++L)
            {
                Li = L*n_conf_states + i;
                if(R_K_i[Li]>SMALL) {
                    for(K=0; K<n_therm_states; ++K)
                    {
                        Ki = K*n_conf_states + i;
                        KLi = K*n_therm_states*n_conf_states + L*n_conf_states + i;
                        skipped_biased_conf_weights[KLi] += bias_sequence[K*seq_length + x]/skipped_divisor[L];
                    }
                }
            }
        }
    }

    /* major arguments for this approach: 
     * - leaving out thermodynamic states leads to an increase ot z_K_i
     *   (no cancellation of negative and positive effects) so the 
     *   effect of leaving out a state can never go undetected.
     * - effect of R_K_i=0 (disconnectivity is known); can ask question:
     *   does the small magnitude of the (normalized) Boltzmann weights
     *   leads to the same behaviour?
     * */
    /* generate connectivity matrix */
    if(check_overlap) {
        n = 0;
        for(i=0; i<n_conf_states; ++i)
        {
            for(L=0; L<n_therm_states; ++L)
            {
                Li = L*n_conf_states + i;
                if(R_K_i[Li]>SMALL) {
                    for(K=0; K<n_therm_states; ++K)
                    {
                        Ki = K*n_conf_states + i;
                        KLi = K*n_therm_states*n_conf_states + L*n_conf_states + i;
                        //if(K!=L)
                        //    printf("test[%d,%d,%d] = %g/%g=%g\n",K,L,i,
                        //        fabs(skipped_biased_conf_weights[KLi]-new_biased_conf_weights[Ki]),
                        //        new_biased_conf_weights[Ki],
                        //        fabs(skipped_biased_conf_weights[KLi]-new_biased_conf_weights[Ki])/new_biased_conf_weights[Ki]);
                        if(K!=L && fabs(skipped_biased_conf_weights[KLi]-new_biased_conf_weights[Ki])>THRESH*new_biased_conf_weights[Ki]) {
                            //printf("contibution = %f\n", fabs(skipped_biased_conf_weights[KLi]-new_biased_conf_weights[Ki])/new_biased_conf_weights[Ki]);
                            n++;
                        }
                    }
                }
            }
        }

        cols = malloc(n*sizeof(int));
        rows = malloc(n*sizeof(int));
        assert(cols && rows);
        s.cols = cols;
        s.rows = rows;
        s.length = n;

        n = 0;
        for(i=0; i<n_conf_states; ++i)
        {
            for(L=0; L<n_therm_states; ++L)
            {
                Li = L*n_conf_states + i;
                if(R_K_i[Li]>SMALL) {
                    for(K=0; K<n_therm_states; ++K)
                    {
                        Ki = K*n_conf_states + i;
                        KLi = K*n_therm_states*n_conf_states + L*n_conf_states + i;
                        if(K!=L && fabs(skipped_biased_conf_weights[KLi]-new_biased_conf_weights[Ki])>THRESH*new_biased_conf_weights[Ki]) {
                            cols[n] = Ki;
                            rows[n] = Li;
                            n++;
                        }
                    }
                }
            }
        }

        free(skipped_biased_conf_weights);
        free(skipped_divisor);
    }

    /* normalization */
    sum = 0.0;
    for(i=0; i < n_conf_states * n_therm_states; ++i) sum += new_biased_conf_weights[i];
    for(i=0; i < n_conf_states * n_therm_states; ++i) new_biased_conf_weights[i] = new_biased_conf_weights[i]/sum;
    
    return s;
}
