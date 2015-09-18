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

extern void _set_lognu(
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

extern void _iterate_lognu(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij,
    int n_therm_states, int n_markov_states, double *scratch_M, double *new_log_nu_K_i)
{
    int i, j, K, o;
    int MM=n_markov_states*n_markov_states, Ki, Kj;
    int CK, CKij, CKji;
    double divisor;
    for(i=0; i<n_markov_states; ++i)
    {
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_markov_states+i;
            o = 0;
            for(j=0; j<n_markov_states; ++j)
            {
                CKij = C_K_ij[K*MM + i*n_markov_states + j];
                CKji = C_K_ij[K*MM + j*n_markov_states + i];
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_M[o++] = (0 == CKij) ?
                        THERMOTOOLS_DTRAM_LOG_PRIOR : log(THERMOTOOLS_DTRAM_PRIOR + (double) CKij);
                    continue;
                }
                CK = CKij + CKji;
                Kj = K*n_markov_states+j;
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                divisor = _logsumexp_pair(
                        log_nu_K_i[Kj] - f_i[i] - b_K_i[Ki], log_nu_K_i[Ki] - f_i[j] - b_K_i[Kj]);
                scratch_M[o++] = log((double) CK) - b_K_i[Kj] - f_i[j] + log_nu_K_i[Ki] - divisor;
            }
            new_log_nu_K_i[Ki] = _logsumexp(scratch_M, o);
        }
    }
}

extern void _iterate_fi(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
    int n_markov_states, double *scratch_TM, double *scratch_M, double *new_f_i)
{
    int i, j, K, o;
    int MM=n_markov_states*n_markov_states, KM=n_therm_states*n_markov_states, Ki, Kj;
    int CK, CKij, CKji, Ci;
    double divisor, norm;
    for(i=0; i<n_markov_states; ++i)
    {
        Ci = 0;
        o = 0;
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
                    scratch_TM[o] = (0 == CKij) ?
                        THERMOTOOLS_DTRAM_LOG_PRIOR : log(THERMOTOOLS_DTRAM_PRIOR + (double) CKij);
                    scratch_TM[o++] += f_i[i];
                    continue;
                }
                CK = CKij + CKji;
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                divisor = _logsumexp_pair(
                        log_nu_K_i[Kj] - f_i[i] - b_K_i[Ki], log_nu_K_i[Ki] - f_i[j] - b_K_i[Kj]);
                scratch_TM[o++] = log((double) CK) - b_K_i[Ki] + log_nu_K_i[Kj] - divisor;
            }
        }
        /* patch Ci and the total divisor together */
        new_f_i[i] = _logsumexp(scratch_TM, o) - log(
            n_therm_states*THERMOTOOLS_DTRAM_PRIOR + (double) Ci);
        scratch_M[i] = -new_f_i[i];
    }
    norm = _logsumexp(scratch_M, n_markov_states);
    for(i=0; i<n_markov_states; ++i)
        new_f_i[i] += norm;
}

extern void _get_p(
    double *log_nu_i, double *b_i, double *f_i, int *C_ij,
    int n_markov_states, double *scratch_M, double *p_ij)
{
    int i, j, o;
    int ij, ji;
    int C;
    double divisor, sum;
    for(i=0; i<n_markov_states; ++i)
    {
        o = 0;
        for(j=0; j<n_markov_states; ++j)
        {
            ij = i*n_markov_states + j;
            p_ij[ij] = 0.0;
            /* special case: diagonal element */
            if(i == j)
            {
                scratch_M[o] = (0 == C_ij[ij]) ?
                    THERMOTOOLS_DTRAM_LOG_PRIOR : log(THERMOTOOLS_DTRAM_PRIOR + (double) C_ij[ij]);
                scratch_M[o] -= log_nu_i[i];
                p_ij[ij] = exp(scratch_M[o++]);
                continue;
            }
            ji = j*n_markov_states + i;
            C = C_ij[ij] + C_ij[ji];
            /* special case: this element is zero */
            if(0 == C) continue;
            /* regular case */
            divisor = _logsumexp_pair(
                    log_nu_i[j] - f_i[i] - b_i[i], log_nu_i[i] - f_i[j] - b_i[j]);
            scratch_M[o] =  log((double) C) - f_i[j] - b_i[j] - divisor;
            p_ij[ij] = exp(scratch_M[o++]);
        }
        /* compute the diagonal elements from the other elements in this line */
        sum = exp(_logsumexp(scratch_M, o));
        if(0.0 == sum)
        {
            for(j=0; j<n_markov_states; ++j)
                p_ij[i*n_markov_states + j] = 0.0;
            p_ij[i*n_markov_states + i] = 1.0;
        }
        else if(1.0 != sum)
        {
            for(j=0; j<n_markov_states; ++j)
                p_ij[i*n_markov_states + j] /= sum;
        }
    }
}

extern void _get_fk(
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
