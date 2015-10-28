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
    int n_therm_states, int n_conf_states, int iteration, double *new_lagrangian_mult)
{
    int i, j, K, KMM, Ki, Kj;
    int CCT_Kij;
    double sum;

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
                        //if(lagrangian_mult[Ki]==0) { fprintf(stderr, "Warning nu[%d,%d]=0 in iteration %d.\n",K,i,iteration); }
                        //if(lagrangian_mult[Kj]==0) { fprintf(stderr, "Warning nu[%d,%d]=0 in iteration %d.\n",K,j,iteration); }
                        if(lagrangian_mult[Ki]==0 && lagrangian_mult[Kj]==0) fprintf(stderr, "Warning nu[%d,%d]=nu[%d,%d]=0 in iteration %d.\n",K,i,K,j,iteration);
                        if(lagrangian_mult[Ki]+lagrangian_mult[Kj]<CCT_Kij) fprintf(stderr, "Not a valid nu iterate at K=%d, i=%d, j=%d in iteration %d.\n", K,i,j,iteration);
                        //if(biased_conf_weights[Ki]==0) fprintf(stderr, "Warning Z[%d,%d]=0 in iteration %d.\n",K,i,iteration);
                        //if(biased_conf_weights[Kj]==0) fprintf(stderr, "Warning Z[%d,%d]=0 in iteration %d.\n",K,j,iteration);
                        if(fabs(log(biased_conf_weights[Ki])-log(biased_conf_weights[Kj])) > 30) {
                            fprintf(stderr, "Warning unrealistic free energy difference between Z[%d,%d] and Z[%d,%d] in iteration %d.\n",K,i,K,j,iteration);
                        }
                        new_lagrangian_mult[Ki] += 
                            (double)CCT_Kij / (1.0 + 
                                               (lagrangian_mult[Kj]/lagrangian_mult[Ki])*
                                               (biased_conf_weights[Ki]/biased_conf_weights[Kj]));
                        if(new_lagrangian_mult[Ki]< 1.E-300) {  
                            //fprintf(stderr, "Warning new_nu[%d,%d]=0 in iteration %d. fixed.\n",K,i,iteration);
                            new_lagrangian_mult[Ki] =1.E-300; 
                        }
                    }
                }
                //if(lagrangian_mult[Ki] > 0 && new_lagrangian_mult[Ki] > 0)
                //    fprintf(stderr, "nu[%d,%d] was scaled by %g in iteration %d.\n", K,i,new_lagrangian_mult[Ki]/lagrangian_mult[Ki], iteration);
            }
        }
    }

    for(i=0; i<n_conf_states; ++i)
    {
        sum = 0;
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            sum += lagrangian_mult[Ki];
        }
        if(sum < 1.E-50) { fprintf(stderr, "All Lagrange multipliers for i=%d are zero.\n", i); }
    }


}

struct my_sparse _update_biased_conf_weights(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, double *bias_sequence,
    int *state_sequence, int *state_counts, int seq_length, double *R_K_i,
    int n_therm_states, int n_conf_states, int check_overlap, double *scratch_TM, double *new_biased_conf_weights)
{
    int i, j, K, Ki, Kj, KMM, x, n, L, Li, KLi;
    int CCT_Kij, CKi, CtKi;
    double divisor, sum, term, c, t;
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
            CtKi = 0;
            Ki = K*n_conf_states + i;
            R_K_i[Ki] = 0.0;
            for(j=0; j<n_conf_states; ++j)
            {
                CCT_Kij = count_matrices[KMM + i*n_conf_states + j];
                CKi += count_matrices[KMM + j*n_conf_states + i];
                CtKi += count_matrices[KMM + i*n_conf_states + j];
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

    /* actual update */
    for(i=0; i < n_conf_states*n_therm_states; ++i) new_biased_conf_weights[i] = 0.0;
    for(i=0; i < n_conf_states*n_therm_states; ++i) scratch_TM[i] = 0.0;
    for(x=0; x < seq_length; ++x) 
    {
        i = state_sequence[x];
        assert(i<n_conf_states);
        divisor = 0;
        c = 0;
        /* calulate normal divisor */
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            if(THERMOTOOLS_TRAM_PRIOR < R_K_i[Ki]) { 
                term = R_K_i[Ki]*bias_sequence[K*seq_length + x]/biased_conf_weights[Ki] - c;
                t = divisor + term;
                c = (t - divisor) - term;
                divisor = t;
                //divisor += term; 
                //if(term>0 && fabs(term)<1.E-14*fabs(divisor)) { fprintf(stderr, "Warning: small element.\n"); }
                
                
            }
        }
        if(divisor==0) fprintf(stderr, "divisor is zero. should never happen!\n");
        if(isnan(divisor)) fprintf(stderr, "divisor is NaN. should never happen!\n");

        /* update normal weights */
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            term = bias_sequence[K*seq_length + x]/divisor - scratch_TM[Ki];
            t = new_biased_conf_weights[Ki] + term;
            scratch_TM[Ki] = (t - new_biased_conf_weights[Ki]) - term;
            new_biased_conf_weights[Ki] = t;
            if(isnan(new_biased_conf_weights[Ki])) { fprintf(stderr, "Z:Warning Z[%d,%d]=NaN (%f,%f) %d\n",K,i,bias_sequence[K*seq_length + x],divisor,x); exit(1); }
        }

    }

    /* normalization */
    //sum = 0.0;
    //for(i=0; i < n_conf_states * n_therm_states; ++i) sum += new_biased_conf_weights[i];
    //for(i=0; i < n_conf_states * n_therm_states; ++i) new_biased_conf_weights[i] = new_biased_conf_weights[i]/sum;
    
    return s;
}
