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
#include "_tram.h"

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

void _set_lognu(double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states)
{
    int i, j, K;
    int MM = n_markov_states * n_markov_states, KMM;
    int sum;
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K * MM;
        for(i=0; i<n_markov_states; ++i)
        {
            sum = 0;
            for(j=0; j<n_markov_states; ++j)
                sum += C_K_ij[KMM + i * n_markov_states + j];
            log_nu_K_i[K * n_markov_states + i] = log(THERMOTOOLS_TRAM_PRIOR + sum);
        }
    }

}

void _iterate_lognu(
    double *log_nu_K_i, double *f_K_i, int *C_K_ij,
    int n_therm_states, int n_markov_states, double *scratch_M, double *new_log_nu_K_i)
{
    int i, j, K, o;
    int MM = n_markov_states * n_markov_states, Ki, Kj, KM, KMM;
    int CK, CKij;
    double divisor;
    for(K=0; K<n_therm_states; ++K)
    {
        KM = K * n_markov_states;
        KMM = KM * n_markov_states;
        for(i=0; i<n_markov_states; ++i)
        {
            Ki = KM + i;
            o = 0;
            for(j=0; j<n_markov_states; ++j)
            {
                CKij = C_K_ij[KMM + i * n_markov_states + j];
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_M[o++] = (0 == CKij) ?
                        THERMOTOOLS_TRAM_LOG_PRIOR : log(THERMOTOOLS_TRAM_PRIOR + (double) CKij);
                    continue;
                }
                CK = CKij + C_K_ij[KMM + j * n_markov_states + i];
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                Kj = KM + j;
                divisor = _logsumexp_pair(log_nu_K_i[Kj] - f_K_i[Ki], log_nu_K_i[Ki] - f_K_i[Kj]);
                scratch_M[o++] = log((double) CK) + log_nu_K_i[Ki] - f_K_i[Kj] - divisor;
            }
            new_log_nu_K_i[Ki] = _logsumexp(scratch_M, o);
        }
    }
}

void _iterate_fki(
    double *log_nu_K_i, double *f_K_i, int *C_K_ij, double *b_K_x,
    int *M_x, int *N_K_i, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_markov_states, double *scratch_M, double *scratch_T,
    double *new_f_K_i)
{
    int i, j, K, x, o;
    int Ki, Kj, KM, KMM;
    int Ci, CK, CKij, CKji, NC;
    double divisor, R_addon, shift;
    /* compute R_K_i */
    for(K=0; K<n_therm_states; ++K)
    {
        KM = K * n_markov_states;
        KMM = KM * n_markov_states;
        for(i=0; i<n_markov_states; ++i)
        {
            Ci = 0;
            Ki = KM + i;
            o = 0;
            for(j=0; j<n_markov_states; ++j)
            {
                CKij = C_K_ij[KMM + i * n_markov_states + j];
                CKji = C_K_ij[KMM + j * n_markov_states + i];
                Ci += CKji;
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_M[o] = (0 == CKij) ? THERMOTOOLS_TRAM_LOG_PRIOR : log(THERMOTOOLS_TRAM_PRIOR + (double) CKij);
                    scratch_M[o++] += f_K_i[Ki];
                    continue;
                }
                CK = CKij + CKji;
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                Kj = KM + j;
                divisor = _logsumexp_pair(log_nu_K_i[Kj] - f_K_i[Ki], log_nu_K_i[Ki] - f_K_i[Kj]);
                scratch_M[o++] = log((double) CK) + log_nu_K_i[Kj] - divisor;
            }
            NC = N_K_i[Ki] - Ci;
            R_addon = (0 < NC) ? log((double) NC) + f_K_i[Ki] : -INFINITY; /* IGNORE PRIOR */
            log_R_K_i[Ki] = _logsumexp_pair(_logsumexp(scratch_M, o), R_addon);
        }
    }
    /* set new_f_K_i to infinity (z_K_i==0) */
    KM = n_therm_states * n_markov_states;
    for(i=0; i<KM; ++i)
        new_f_K_i[i] = INFINITY;
    /* compute new f_K_i */
    for(x=0; x<seq_length; ++x)
    {
        i = M_x[x];
        for(K=0; K<n_therm_states; ++K) /* TODO: use continue for R_K_i=0 cases */
            scratch_T[K] = log_R_K_i[K * n_markov_states + i] - b_K_x[K * seq_length + x];
        divisor = _logsumexp(scratch_T, n_therm_states);
        for(K=0; K<n_therm_states; ++K)
        {
            new_f_K_i[K * n_markov_states + i] = -_logsumexp_pair(
                    -new_f_K_i[K * n_markov_states + i], -(divisor + b_K_x[K * seq_length + x]));
        }
    }
    /* prevent drift */
    KM = n_therm_states * n_markov_states;
    shift = new_f_K_i[0];
    for(i=1; i<KM; ++i)
        shift = (shift < new_f_K_i[i]) ? shift : new_f_K_i[i];
    for(i=0; i<KM; ++i)
        new_f_K_i[i] -= shift;
}

void _get_fi(
    double *b_K_x, int *M_x, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_markov_states, double *scratch_M, double *scratch_T,
    double *f_i)
{
    int i, K, x;
    double divisor, norm;

    /* set f_i to infinity (pi_i==0) */
    for(i=0; i<n_markov_states; ++i)
        f_i[i] = INFINITY;
    
    /* compute new f_i */
    for( x=0; x<seq_length; ++x )
    {
        i = M_x[x];
        for(K=0; K<n_therm_states; ++K)
            scratch_T[K] = log_R_K_i[K * n_markov_states + i] - b_K_x[K * seq_length + x];
        divisor = _logsumexp(scratch_T, n_therm_states);
        f_i[i] = -_logsumexp_pair(-f_i[i], -divisor);
    }
}

void _normalize_fki(
    double *f_i, double *f_K_i, int n_therm_states, int n_markov_states, double *scratch_M)
{
    int i, KM = n_therm_states * n_markov_states;
    double f0;
    for(i=0; i<n_markov_states; ++i)
        scratch_M[i] = -f_i[i];
    f0 = -_logsumexp(scratch_M, n_markov_states);
    for(i=0; i<n_markov_states; ++i)
        f_i[i] -= f0;
    for(i=0; i<KM; ++i)
        f_K_i[i] -= f0;
}

void _get_p(
    double *log_nu_i, double *f_i, int *C_ij,
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
                    THERMOTOOLS_TRAM_LOG_PRIOR : log(THERMOTOOLS_TRAM_PRIOR + (double) C_ij[ij]);
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
                    log_nu_i[j] - f_i[i], log_nu_i[i] - f_i[j]);
            scratch_M[o] =  log((double) C) - f_i[j] - divisor;
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
