/*

    _dtram.c - dTRAM implementation in C

    author: Christoph Wehmeyer <christoph.wehmeyer@fu-berlin.de>

*/

#include "_dtram.h"

// old m$ visual studio is not c99 compliant (vs2010 eg. is not)
#ifdef _MSC_VER
	#include <math.h>
	#include <float.h>
	#define INFINITY (DBL_MAX+DBL_MAX)
	#define NAN (INFINITY-INFINITY)
#endif


void _log_nu_K_i_setter(
    double *log_nu_K_i,
    int *C_K_ij,
    int n_therm_states,
    int n_markov_states
)
{
    int i, j, K;
    int MM = n_markov_states * n_markov_states, KMM;
    int sum;
    for( K=0; K<n_therm_states; ++K )
    {
        KMM = K * MM;
        for( i=0; i<n_markov_states; ++i )
        {
            sum = 0;
            for( j=0; j<n_markov_states; ++j )
                sum += C_K_ij[ KMM + i*n_markov_states + j ];
            log_nu_K_i[ K * n_markov_states + i ] = log( PYTRAM_DTRAM_PRIOR + (double) sum );
        }
    }
}

void _log_nu_K_i_equation(
    double *log_nu_K_i,
    double *b_K_i,
    double *f_i,
    int *C_K_ij,
    int n_therm_states,
    int n_markov_states,
    double *scratch_j,
    double *new_log_nu_K_i
)
{
    int i, j, K;
    int MM = n_markov_states * n_markov_states, Ki, Kj;
    int CK, CKij, CKji;
    double divisor;
    for( i=0; i<n_markov_states; ++i )
    {
        for( K=0; K<n_therm_states; ++K )
        {
            Ki = K*n_markov_states+i;
            for( j=0; j<n_markov_states; ++j )
            {
                CKij = C_K_ij[K*MM+i*n_markov_states+j];
                CKji = C_K_ij[K*MM+j*n_markov_states+i];
                /* special case: most variables cancel out, here */
                if( i == j )
                {
                    scratch_j[j] = ( 0 == CKij ) ? PYTRAM_DTRAM_LOG_PRIOR : log( PYTRAM_DTRAM_PRIOR + (double) CKij );
                    continue;
                }
                CK = CKij + CKji;
                Kj = K*n_markov_states+j;
                /* special case */
                if( 0 == CK )
                {
                    if( ( -INFINITY == log_nu_K_i[Ki] ) && ( -INFINITY == log_nu_K_i[Kj] ) )
                    {
                        scratch_j[j] = -_logsumexp_pair( 0.0, f_i[j] - f_i[i] + b_K_i[Kj] - b_K_i[Ki] );
                        printf( "####### WARNING ####### scratch_j=%f\n", scratch_j[j] );
                    }
                    else
                        scratch_j[j] = -INFINITY;
                    continue;
                }
                /* regular case */
                divisor = _logsumexp_pair(
                        log_nu_K_i[Kj] - f_i[i] - b_K_i[Ki],
                        log_nu_K_i[Ki] - f_i[j] - b_K_i[Kj]
                    );
                scratch_j[j] = log( (double) CK ) - b_K_i[Kj] - f_i[j] + log_nu_K_i[Ki] - divisor;
            }
            new_log_nu_K_i[Ki] = _logsumexp( scratch_j, n_markov_states );
        }
    }
}

void _f_i_equation(
    double *log_nu_K_i,
    double *b_K_i,
    double *f_i,
    int *C_K_ij,
    int n_therm_states,
    int n_markov_states,
    double *scratch_K_j,
    double *scratch_j,
    double *new_f_i
)
{
    int i, j, K;
    int MM = n_markov_states * n_markov_states, KM = n_therm_states * n_markov_states, Ki, Kj;
    int CK, CKij, CKji, Ci;
    double divisor, norm;
    for( i=0; i<n_markov_states; ++i )
    {
        Ci = 0;
        for( K=0; K<n_therm_states; ++K )
        {
            Ki = K * n_markov_states + i;
            for( j=0; j<n_markov_states; ++j )
            {
                Kj = K * n_markov_states + j;
                CKij = C_K_ij[K*MM+i*n_markov_states+j];
                CKji = C_K_ij[K*MM+j*n_markov_states+i];
                /* add counts to Ci */
                Ci += CKji;
                /* special case: most variables cancel out, here */
                if( i == j )
                {
                    scratch_K_j[Kj] = ( 0 == CKij ) ? PYTRAM_DTRAM_LOG_PRIOR : log( PYTRAM_DTRAM_PRIOR + (double) CKij );
                    scratch_K_j[Kj] += f_i[i];
                    continue;
                }
                CK = CKij + CKji;
                /* special case */
                if( 0 == CK )
                {
                    scratch_K_j[Kj] = -INFINITY;
                    continue;
                }
                /* special case */ /* NaNs possible! CHECK THIS */
                /*if( -INFINITY == log_nu_K_i[Ki] )
                    continue;*/
                /* regular case */
                divisor = _logsumexp_pair(
                        log_nu_K_i[Kj] - f_i[i] - b_K_i[Ki],
                        log_nu_K_i[Ki] - f_i[j] - b_K_i[Kj]
                    );
                scratch_K_j[Kj] = log( (double) CK ) - b_K_i[Ki] + log_nu_K_i[Kj] - divisor;
            }
        }
        /* patch Ci and the total divisor together */
        new_f_i[i] = _logsumexp( scratch_K_j, KM ) - log( n_therm_states * PYTRAM_DTRAM_PRIOR + (double) Ci );
        scratch_j[i] = -new_f_i[i];
    }
    norm = _logsumexp( scratch_j, n_markov_states );
    for( i=0; i<n_markov_states; ++i )
        new_f_i[i] += norm;
}

void _p_K_ij_equation(
    double *log_nu_K_i,
    double *b_K_i,
    double *f_i,
    int *C_K_ij,
    int n_therm_states,
    int n_markov_states,
    double *scratch_j,
    double *p_K_ij
)
{
    int i, j, K;
    int MM = n_markov_states * n_markov_states, KMM, Ki, Kj, ij, ji;
    int CK;
    double divisor, pKi, sum;
    for( K=0; K<n_therm_states; ++K )
    {
        KMM = K*MM;
        for( i=0; i<n_markov_states; ++i )
        {
            pKi = 0.0;
            Ki = K*n_markov_states+i;
            for( j=0; j<n_markov_states; ++j )
            {
                /* special case: we compute the diagonal elements later */
                if( i == j )
                {
                    scratch_j[j] = -INFINITY;
                    continue;
                }
                ij = i*n_markov_states+j;
                ji = j*n_markov_states+i;
                p_K_ij[KMM+ij] = 0.0;
                CK = C_K_ij[KMM+ij] + C_K_ij[KMM+ji];
                /* special case: this element is zero */
                if( 0 == CK )
                {
                    scratch_j[j] = -INFINITY;
                    continue;
                }
                /* regular case */
                Kj = K*n_markov_states+j;
                divisor = _logsumexp_pair(
                        log_nu_K_i[Kj] - f_i[i] - b_K_i[Ki],
                        log_nu_K_i[Ki] - f_i[j] - b_K_i[Kj]
                    );
                scratch_j[j] =  log( (double) CK ) - f_i[j] - b_K_i[Kj] - divisor;
                p_K_ij[KMM+ij] = exp( scratch_j[j] );
            }
            /* compute the diagonal elements from the other elements in this line */
            sum = exp( _logsumexp( scratch_j, n_markov_states ) );
            if( 1.0 <= sum )
            {
                p_K_ij[KMM+i*n_markov_states+i] = 0.0;
                for( j=0; j<n_markov_states; ++j )
                    p_K_ij[KMM+i*n_markov_states+j] /= sum;
            }
            else
            {
                p_K_ij[KMM+i*n_markov_states+i] = 1.0 - sum;
            }
        }       
    }
}


void _f_K_equation(
    double *b_K_i,
    double *f_i,
    int n_therm_states,
    int n_markov_states,
    double *scratch_j,
    double *f_K
)
{
    int K, i;
    for( K=0; K<n_therm_states; ++K )
    {
        for( i=0; i<n_markov_states; ++i )
            scratch_j[i] = -( b_K_i[K*n_markov_states+i] + f_i[i] );
        f_K[K] = _logsumexp( scratch_j, n_markov_states );
    }
}
