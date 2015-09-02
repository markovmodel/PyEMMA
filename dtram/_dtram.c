/*
* This file is part of thermotools.
*
* Copyright 2015 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
*
* thermotools is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <math.h>
#include "../lse/_lse.h"
#include "_dtram.h"

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

extern void _dtram_set_lognu(
    double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states)
{
    int i, j, K;
    int MM=n_markov_states*n_markov_states, KMM;
    int sum;
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K*MM;
        for(i=0; i<n_markov_states; ++i)
        {
            sum = 0;
            for(j=0; j<n_markov_states; ++j)
                sum += C_K_ij[KMM + i*n_markov_states + j];
            log_nu_K_i[K*n_markov_states + i] = log(THERMOTOOLS_DTRAM_PRIOR + (double) sum);
        }
    }
}

extern void _dtram_lognu(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij,
    int n_therm_states, int n_markov_states, double *scratch_M, double *new_log_nu_K_i)
{
    int i, j, K;
    int MM=n_markov_states*n_markov_states, Ki, Kj;
    int CK, CKij, CKji;
    double divisor;
    for(i=0; i<n_markov_states; ++i)
    {
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_markov_states+i;
            for(j=0; j<n_markov_states; ++j)
            {
                CKij = C_K_ij[K*MM + i*n_markov_states + j];
                CKji = C_K_ij[K*MM + j*n_markov_states + i];
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_M[j] = (0 == CKij) ?
                        THERMOTOOLS_DTRAM_LOG_PRIOR : log(THERMOTOOLS_DTRAM_PRIOR + (double) CKij);
                    continue;
                }
                CK = CKij + CKji;
                Kj = K*n_markov_states+j;
                /* special case */
                if(0 == CK)
                {
                    if((-INFINITY == log_nu_K_i[Ki]) && (-INFINITY == log_nu_K_i[Kj]))
                    {
                        scratch_M[j] = -_logsumexp_pair(
                            0.0, f_i[j] - f_i[i] + b_K_i[Kj] - b_K_i[Ki]);
                    }
                    else
                        scratch_M[j] = -INFINITY;
                    continue;
                }
                /* regular case */
                divisor = _logsumexp_pair(
                        log_nu_K_i[Kj] - f_i[i] - b_K_i[Ki], log_nu_K_i[Ki] - f_i[j] - b_K_i[Kj]);
                scratch_M[j] = log((double) CK) - b_K_i[Kj] - f_i[j] + log_nu_K_i[Ki] - divisor;
            }
            new_log_nu_K_i[Ki] = _logsumexp(scratch_M, n_markov_states);
        }
    }
}

extern void _dtram_fi(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
    int n_markov_states, double *scratch_TM, double *scratch_M, double *new_f_i)
{
    int i, j, K;
    int MM=n_markov_states*n_markov_states, KM=n_therm_states*n_markov_states, Ki, Kj;
    int CK, CKij, CKji, Ci;
    double divisor, norm;
    for(i=0; i<n_markov_states; ++i)
    {
        Ci = 0;
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_markov_states + i;
            for(j=0; j<n_markov_states; ++j)
            {
                Kj = K*n_markov_states + j;
                CKij = C_K_ij[K*MM + i*n_markov_states + j];
                CKji = C_K_ij[K*MM + j*n_markov_states + i];
                /* add counts to Ci */
                Ci += CKji;
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_TM[Kj] = (0 == CKij) ?
                        THERMOTOOLS_DTRAM_LOG_PRIOR : log(THERMOTOOLS_DTRAM_PRIOR + (double) CKij);
                    scratch_TM[Kj] += f_i[i];
                    continue;
                }
                CK = CKij + CKji;
                /* special case */
                if(0 == CK)
                {
                    scratch_TM[Kj] = -INFINITY;
                    continue;
                }
                /* special case */ /* NaNs possible! CHECK THIS */
                /*if( -INFINITY == log_nu_K_i[Ki] )
                    continue;*/
                /* regular case */
                divisor = _logsumexp_pair(
                        log_nu_K_i[Kj] - f_i[i] - b_K_i[Ki], log_nu_K_i[Ki] - f_i[j] - b_K_i[Kj]);
                scratch_TM[Kj] = log((double) CK) - b_K_i[Ki] + log_nu_K_i[Kj] - divisor;
            }
        }
        /* patch Ci and the total divisor together */
        new_f_i[i] = _logsumexp(scratch_TM, KM) - log(
            n_therm_states*THERMOTOOLS_DTRAM_PRIOR + (double) Ci);
        scratch_M[i] = -new_f_i[i];
    }
    norm = _logsumexp(scratch_M, n_markov_states);
    for(i=0; i<n_markov_states; ++i)
        new_f_i[i] += norm;
}

extern void _dtram_pk(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
    int n_markov_states, double *scratch_M, double *p_K_ij)
{
    int i, j, K;
    int MM=n_markov_states*n_markov_states, KMM, Ki, Kj, ij, ji;
    int CK;
    double divisor, sum;
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K*MM;
        for(i=0; i<n_markov_states; ++i)
        {
            Ki = K*n_markov_states + i;
            for(j=0; j<n_markov_states; ++j)
            {
                /* special case: we compute the diagonal elements later */
                if(i == j)
                {
                    scratch_M[j] = -INFINITY;
                    continue;
                }
                ij = i*n_markov_states + j;
                ji = j*n_markov_states + i;
                p_K_ij[KMM + ij] = 0.0;
                CK = C_K_ij[KMM + ij] + C_K_ij[KMM + ji];
                /* special case: this element is zero */
                if(0 == CK)
                {
                    scratch_M[j] = -INFINITY;
                    continue;
                }
                /* regular case */
                Kj = K*n_markov_states + j;
                divisor = _logsumexp_pair(
                        log_nu_K_i[Kj] - f_i[i] - b_K_i[Ki], log_nu_K_i[Ki] - f_i[j] - b_K_i[Kj]);
                scratch_M[j] =  log((double) CK) - f_i[j] - b_K_i[Kj] - divisor;
                p_K_ij[KMM + ij] = exp(scratch_M[j]);
            }
            /* compute the diagonal elements from the other elements in this line */
            sum = exp(_logsumexp(scratch_M, n_markov_states));
            if(1.0 <= sum)
            {
                p_K_ij[KMM + i*n_markov_states + i] = 0.0;
                for(j=0; j<n_markov_states; ++j)
                    p_K_ij[KMM + i*n_markov_states + j] /= sum;
            }
            else
            {
                p_K_ij[KMM + i*n_markov_states + i] = 1.0 - sum;
            }
        }       
    }
}

extern void _dtram_p(
    double *log_nu_i, double *b_i, double *f_i, int *C_ij,
    int n_markov_states, double *scratch_M, double *p_ij)
{
    int i, j;
    int ij, ji;
    int C;
    double divisor, sum;
    for(i=0; i<n_markov_states; ++i)
    {
        for(j=0; j<n_markov_states; ++j)
        {
            /* special case: we compute the diagonal elements later */
            if(i == j)
            {
                scratch_M[j] = -INFINITY;
                continue;
            }
            ij = i*n_markov_states + j;
            ji = j*n_markov_states + i;
            p_ij[ij] = 0.0;
            C = C_ij[ij] + C_ij[ji];
            /* special case: this element is zero */
            if(0 == C)
            {
                scratch_M[j] = -INFINITY;
                continue;
            }
            /* regular case */
            divisor = _logsumexp_pair(
                    log_nu_i[j] - f_i[i] - b_i[i], log_nu_i[i] - f_i[j] - b_i[j]);
            scratch_M[j] =  log((double) C) - f_i[j] - b_i[j] - divisor;
            p_ij[ij] = exp(scratch_M[j]);
        }
        /* compute the diagonal elements from the other elements in this line */
        sum = exp(_logsumexp(scratch_M, n_markov_states));
        if(1.0 <= sum)
        {
            p_ij[i*n_markov_states + i] = 0.0;
            for(j=0; j<n_markov_states; ++j)
                p_ij[i*n_markov_states + j] /= sum;
        }
        else
        {
            p_ij[i*n_markov_states + i] = 1.0 - sum;
        }
    }
}

extern void _dtram_fk(
    double *b_K_i, double *f_i, int n_therm_states, int n_markov_states,
    double *scratch_M, double *f_K)
{
    int K, i;
    for(K=0; K<n_therm_states; ++K)
    {
        for(i=0; i<n_markov_states; ++i)
            scratch_M[i] = -(b_K_i[K*n_markov_states + i] + f_i[i]);
        f_K[K] = -_logsumexp(scratch_M, n_markov_states);
    }
}
