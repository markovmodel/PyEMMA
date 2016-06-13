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

#include <stdio.h>
#include <assert.h>

#include "_tram.h"
#include "../util/_util.h"

void _tram_init_lagrangian_mult(int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult)
{
    int i, j, K;
    int MM = n_conf_states * n_conf_states, KMM;
    double sum;
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K * MM;
        for(i=0; i<n_conf_states; ++i)
        {
            sum = 0.0;
            for(j=0; j<n_conf_states; ++j)
                sum += 0.5 * (count_matrices[KMM + i * n_conf_states + j]+
                              count_matrices[KMM + j * n_conf_states + i]);
            log_lagrangian_mult[K * n_conf_states + i] = log(THERMOTOOLS_TRAM_PRIOR + sum);
        }
    }

}

void _tram_update_lagrangian_mult(
    double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices, int* state_counts,
    int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult)
{
    int i, j, K, o;
    int Ki, Kj, KM, KMM;
    int CK, CKij;
    double divisor;
    for(K=0; K<n_therm_states; ++K)
    {
        KM = K * n_conf_states;
        KMM = KM * n_conf_states;
        for(i=0; i<n_conf_states; ++i)
        {
            Ki = KM + i;
            if(0 == state_counts[Ki])
            {
                new_log_lagrangian_mult[Ki] = -INFINITY;
                continue;
            }            
            o = 0;
            for(j=0; j<n_conf_states; ++j)
            {
                CKij = count_matrices[KMM + i * n_conf_states + j];
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_M[o++] = (0 == CKij) ?
                        THERMOTOOLS_TRAM_LOG_PRIOR : log(THERMOTOOLS_TRAM_PRIOR + (double) CKij);
                    continue;
                }
                CK = CKij + count_matrices[KMM + j * n_conf_states + i];
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                Kj = KM + j;
                divisor = _logsumexp_pair(
                    log_lagrangian_mult[Kj] - biased_conf_energies[Ki] - log_lagrangian_mult[Ki] + biased_conf_energies[Kj], 0.0);
                scratch_M[o++] = log((double) CK) - divisor;
            }
            new_log_lagrangian_mult[Ki] = _logsumexp_sort_kahan_inplace(scratch_M, o);
        }
    }
}

void _tram_get_log_Ref_K_i(
    double *log_lagrangian_mult, double *biased_conf_energies,
    int *count_matrices, int *state_counts,
    int n_therm_states, int n_conf_states, double *scratch_M, double *log_R_K_i
#ifdef TRAMMBAR
    ,
    double *therm_energies, int *equilibrium_therm_state_counts,
    double overcounting_factor
#endif
)
{
    int i, j, K, o;
    int Ki, Kj, KM, KMM;
    int Ci, CK, CKij, CKji, NC;
    double divisor, R_addon;

    for(K=0; K<n_therm_states; ++K)
    {
        KM = K * n_conf_states;
        KMM = KM * n_conf_states;
        for(i=0; i<n_conf_states; ++i)
        {
            Ki = KM + i;
            if(0 == state_counts[Ki]) /* applying Hao's speed-up recomendation */
            {
                log_R_K_i[Ki] = -INFINITY;
                continue;
            }
            Ci = 0;
            o = 0;
            for(j=0; j<n_conf_states; ++j)
            {
                CKij = count_matrices[KMM + i * n_conf_states + j];
                CKji = count_matrices[KMM + j * n_conf_states + i];
                Ci += CKji;
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_M[o] = (0 == CKij) ? THERMOTOOLS_TRAM_LOG_PRIOR : log(THERMOTOOLS_TRAM_PRIOR + (double) CKij);
                    scratch_M[o++] += biased_conf_energies[Ki];
                    continue;
                }
                CK = CKij + CKji;
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                Kj = KM + j;
                divisor = _logsumexp_pair(
                    log_lagrangian_mult[Kj] - biased_conf_energies[Ki],
                    log_lagrangian_mult[Ki] - biased_conf_energies[Kj]);
                scratch_M[o++] = log((double) CK) + log_lagrangian_mult[Kj] - divisor;
            }
            NC = state_counts[Ki] - Ci;
            R_addon = (0 < NC) ? log((double) NC) + biased_conf_energies[Ki] : -INFINITY; /* IGNORE PRIOR */
            log_R_K_i[Ki] = _logsumexp_pair(_logsumexp_sort_kahan_inplace(scratch_M, o), R_addon);
        }
    }

#ifdef TRAMMBAR
    if(equilibrium_therm_state_counts && therm_energies)
    {
        for(K=0; K<n_therm_states; ++K)
        {
            KM = K * n_conf_states;
            for(i=0; i<n_conf_states; ++i)
                log_R_K_i[KM + i] += log(overcounting_factor);
        }
        for(K=0; K<n_therm_states; ++K)
        {
            if(0 < equilibrium_therm_state_counts[K])
            {
                KM = K * n_conf_states;
                for(i=0; i<n_conf_states; ++i)
                {
                    Ki = KM + i;
                    log_R_K_i[Ki] = _logsumexp_pair(log_R_K_i[Ki], log(equilibrium_therm_state_counts[K]) + therm_energies[K]);
                }
            }
        }
    }
#endif
}

double _tram_update_biased_conf_energies(
    double *bias_energy_sequence, int *state_sequence, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_T, double *new_biased_conf_energies, int return_log_L)
{
    int i, K, x, o, Ki;
    int KM;
    double divisor, log_L;

    /* assume that new_biased_conf_energies have been set to INF by the caller in the first call */
    for(x=0; x<seq_length; ++x)
    {
        i = state_sequence[x];
        if(i < 0) continue; /* skip frames that have negative Markov state indices */
        o = 0;
        for(K=0; K<n_therm_states; ++K)
        {
            assert(K<n_therm_states);
            assert(K>=0);
            /* applying Hao's speed-up recomendation */
            if(-INFINITY == log_R_K_i[K * n_conf_states + i]) continue;
            scratch_T[o++] = log_R_K_i[K * n_conf_states + i] - bias_energy_sequence[x * n_therm_states + K];
        }
        divisor = _logsumexp_sort_kahan_inplace(scratch_T, o);
        
        for(K=0; K<n_therm_states; ++K)
        {
            new_biased_conf_energies[K * n_conf_states + i] = -_logsumexp_pair(
                    -new_biased_conf_energies[K * n_conf_states + i],
                    -(divisor + bias_energy_sequence[x * n_therm_states + K]));
        }
    }

    if(return_log_L) {
        /* -\sum_{x}\log\sum_{l}R_{i(x)}^{(l)}e^{-b^{(l)}(x)+f_{i(x)}^{(l)}} */
        log_L = 0;
        for(x=0; x<seq_length; ++x) {
            o = 0;
            i = state_sequence[x];
            if(i < 0) continue;
            for(K=0; K<n_therm_states; ++K) {
                KM = K*n_conf_states;
                Ki = KM + i;
                if(log_R_K_i[Ki] > 0)
                    scratch_T[o++] =
                        log_R_K_i[Ki] - bias_energy_sequence[x * n_therm_states + K];
                }
            log_L -= _logsumexp_sort_kahan_inplace(scratch_T,o);
        }
        return log_L;
    } else
        return 0;
}

void _tram_get_conf_energies(
    double *bias_energy_sequence, int *state_sequence, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_T, double *conf_energies)
{
    int i, K, x, o;
    double divisor;
    /* assume that conf_energies was set to INF by the caller on the first call */
    for( x=0; x<seq_length; ++x )
    {
        i = state_sequence[x];
        if(i < 0) continue;
        o = 0;
        for(K=0; K<n_therm_states; ++K) {
            if(-INFINITY == log_R_K_i[K * n_conf_states + i]) continue;
            scratch_T[o++] = log_R_K_i[K * n_conf_states + i] - bias_energy_sequence[x * n_therm_states + K];
        }
        divisor = _logsumexp_sort_kahan_inplace(scratch_T, o);
        conf_energies[i] = -_logsumexp_pair(-conf_energies[i], -divisor);
    }
}

void _tram_get_therm_energies(
    double *biased_conf_energies, int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies)
{
    int i, K;
    for(K=0; K<n_therm_states; ++K)
    {
        for(i=0; i<n_conf_states; ++i)
            scratch_M[i] = -biased_conf_energies[K * n_conf_states + i];
        therm_energies[K] = -_logsumexp_sort_kahan_inplace(scratch_M, n_conf_states);
    }
}

void _tram_normalize(
    double *conf_energies, double *biased_conf_energies, double *therm_energies,
    int n_therm_states, int n_conf_states, double *scratch_M)
{
    int i, KM = n_therm_states * n_conf_states;
    double f0;
    for(i=0; i<n_conf_states; ++i)
        scratch_M[i] = -conf_energies[i];
    f0 = -_logsumexp_sort_kahan_inplace(scratch_M, n_conf_states);
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] -= f0;
    for(i=0; i<KM; ++i)
        biased_conf_energies[i] -= f0;
    for(i=0; i<n_therm_states; ++i)
        therm_energies[i] -= f0;
}

void _tram_estimate_transition_matrix(
    double *log_lagrangian_mult, double *conf_energies, int *count_matrix,
    int n_conf_states, double *scratch_M, double *transition_matrix)
{
    int i, j;
    int ij, ji;
    int C;
    double divisor, max_sum;
    double *sum;
    sum = scratch_M;
    for(i=0; i<n_conf_states; ++i)
    {
        sum[i] = 0.0;
        for(j=0; j<n_conf_states; ++j)
        {
            ij = i*n_conf_states + j;
            ji = j*n_conf_states + i;
            transition_matrix[ij] = 0.0;
            C = count_matrix[ij] + count_matrix[ji];
            /* special case: this element is zero */
            if(0 == C) continue;
            if(i == j) {
                /* special case: diagonal element */
                transition_matrix[ij] = 0.5 * C * exp(-log_lagrangian_mult[i]);
            } else {
                /* regular case */
                divisor = _logsumexp_pair(
                    log_lagrangian_mult[j] - conf_energies[i],
                    log_lagrangian_mult[i] - conf_energies[j]);
                transition_matrix[ij] = C * exp(-(conf_energies[j] + divisor));
            }
            sum[i] += transition_matrix[ij];
        }
    }
    /* normalize T matrix */ /* TODO: unify with util._renormalize_transition_matrix? */
    max_sum = 0;
    for(i=0; i<n_conf_states; ++i) if(sum[i] > max_sum) max_sum = sum[i];
    if(max_sum==0) max_sum = 1.0; /* completely empty T matrix -> generate Id matrix */
    for(i=0; i<n_conf_states; ++i) {
        for(j=0; j<n_conf_states; ++j) {
            if(i==j) {
                transition_matrix[i*n_conf_states + i] = (transition_matrix[i*n_conf_states + i]+max_sum-sum[i])/max_sum;
                if(0 == transition_matrix[i*n_conf_states + i] && 0 < count_matrix[i*n_conf_states + i])
                    fprintf(stderr, "# Warning: zero diagonal element T[%d,%d] with non-zero counts.\n", i, i);
            } else {
                transition_matrix[i*n_conf_states + j] = transition_matrix[i*n_conf_states + j]/max_sum; 
            }
        }
    }

}

/* TRAM log-likelihood that comes from the terms containing discrete quantities */
double _tram_discrete_log_likelihood_lower_bound(
    double *log_lagrangian_mult, double *biased_conf_energies,
    int *count_matrices, int *state_counts,
    int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_MM
#ifdef TRAMMBAR
    ,
    double *therm_energies, int *equilibrium_therm_state_counts,
    double overcounting_factor
#endif
)
{
    double a, b;
    int K, i, j;
    int KM, KMM, Ki;
    int CKij;
    double *T_ij;

    /* \sum_{i,j,k}c_{ij}^{(k)}\log p_{ij}^{(k)} */
    a = 0;
    T_ij = scratch_MM;
    for(K=0; K<n_therm_states; ++K)
    {
        KM = K * n_conf_states;
        KMM = KM * n_conf_states;
        _tram_estimate_transition_matrix(
           &log_lagrangian_mult[KM], &biased_conf_energies[KM], &count_matrices[KMM],
           n_conf_states, scratch_M, T_ij);
        for(i=0; i<n_conf_states; ++i)
        {
            for(j=0; j<n_conf_states; ++j)
            {
                CKij = count_matrices[KMM + i * n_conf_states + j];
                if(0==CKij) continue;
                if(i==j) {
                    a += ((double)CKij + THERMOTOOLS_TRAM_PRIOR) * log(T_ij[i*n_conf_states + j]);
                } else {
                    a += CKij * log(T_ij[i*n_conf_states + j]);
                }
            }
        }
    }

    /* \sum_{i,k}N_{i}^{(k)}f_{i}^{(k)} */
    b = 0;
    for(K=0; K<n_therm_states; ++K) {
        KM = K * n_conf_states;
        for(i=0; i<n_conf_states; ++i) {
            Ki = KM + i;
            if(state_counts[Ki]>0)
                b += (state_counts[Ki] + THERMOTOOLS_TRAM_PRIOR) * biased_conf_energies[Ki];
        }
    }

#ifdef TRAMMBAR
    a *= overcounting_factor;
    b *= overcounting_factor;

    /* \sum_k N_{eq}^{(k)}f^{(k)}*/
    if(equilibrium_therm_state_counts && therm_energies) {
        for(K=0; K<n_therm_states; ++K) {
            if(0 < equilibrium_therm_state_counts[K])
                b += equilibrium_therm_state_counts[K] * therm_energies[K];
        }
    }
#endif

    return a+b;
}

void _tram_get_pointwise_unbiased_free_energies(
    int k, double *bias_energy_sequence, double *therm_energies, int *state_sequence,
    int seq_length, double *log_R_K_i, int n_therm_states, int n_conf_states,
    double *scratch_T, double *pointwise_unbiased_free_energies)
{
    int L, o, i, x;
    double log_divisor;

    for(x=0; x<seq_length; ++x)
    {
        i = state_sequence[x];
        if(i < 0) {
            pointwise_unbiased_free_energies[x] = INFINITY;
            continue;
        }
        o = 0;
        for(L=0; L<n_therm_states; ++L)
        {
            if(-INFINITY == log_R_K_i[L * n_conf_states + i]) continue;
            scratch_T[o++] = log_R_K_i[L * n_conf_states + i] - bias_energy_sequence[x * n_therm_states + L];
        }
        log_divisor = _logsumexp_sort_kahan_inplace(scratch_T, o);
        if(k==-1)
            pointwise_unbiased_free_energies[x] = log_divisor;
        else
            pointwise_unbiased_free_energies[x] = bias_energy_sequence[x * n_therm_states + k] + log_divisor - therm_energies[k];
    }
}
