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
#include <stdio.h>
#include "../lse/_lse.h"
#include "_tram.h"

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

void _init_lagrangian_mult(int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult)
{
    int i, j, K;
    int MM = n_conf_states * n_conf_states, KMM;
    int sum;
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K * MM;
        for(i=0; i<n_conf_states; ++i)
        {
            sum = 0;
            for(j=0; j<n_conf_states; ++j)
                sum += 0.5 * (count_matrices[KMM + i * n_conf_states + j]+
                              count_matrices[KMM + j * n_conf_states + i]);
            log_lagrangian_mult[K * n_conf_states + i] = log(THERMOTOOLS_TRAM_PRIOR + sum);
        }
    }

}

void _update_lagrangian_mult(
    double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices,
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
                divisor = _logsumexp_pair(log_lagrangian_mult[Kj] - biased_conf_energies[Ki] - log_lagrangian_mult[Ki] + biased_conf_energies[Kj], 0.0);
                scratch_M[o++] = log((double) CK) - divisor;
            }
            new_log_lagrangian_mult[Ki] = _logsumexp(scratch_M, o);
        }
    }
}

void _update_biased_conf_energies(
    double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices, double *bias_energy_sequence,
    int *state_sequence, int *state_counts, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_T,
    double *new_biased_conf_energies)
{
    int i, j, K, x, o;
    int Ki, Kj, KM, KMM;
    int Ci, CK, CKij, CKji, NC;
    double divisor, R_addon, shift;
    /* compute R_K_i */
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
                divisor = _logsumexp_pair(log_lagrangian_mult[Kj] - biased_conf_energies[Ki], log_lagrangian_mult[Ki] - biased_conf_energies[Kj]);
                scratch_M[o++] = log((double) CK) + log_lagrangian_mult[Kj] - divisor;
            }
            NC = state_counts[Ki] - Ci;
            R_addon = (0 < NC) ? log((double) NC) + biased_conf_energies[Ki] : -INFINITY; /* IGNORE PRIOR */
            log_R_K_i[Ki] = _logsumexp_pair(_logsumexp(scratch_M, o), R_addon);
        }
    }
    /* set new_biased_conf_energies to infinity (z_K_i==0) */
    KM = n_therm_states * n_conf_states;
    for(i=0; i<KM; ++i)
        new_biased_conf_energies[i] = INFINITY;
    /* compute new biased_conf_energies */
    for(x=0; x<seq_length; ++x)
    {
        i = state_sequence[x];
        o = 0;
        for(K=0; K<n_therm_states; ++K)
        {
            /* applying Hao's speed-up recomendation */
            if(-INFINITY == log_R_K_i[K * n_conf_states + i]) continue;
            scratch_T[o++] = log_R_K_i[K * n_conf_states + i] - bias_energy_sequence[K * seq_length + x];
        }
        divisor = _logsumexp(scratch_T, o);
        for(K=0; K<n_therm_states; ++K)
        {
            new_biased_conf_energies[K * n_conf_states + i] = -_logsumexp_pair(
                    -new_biased_conf_energies[K * n_conf_states + i], -(divisor + bias_energy_sequence[K * seq_length + x]));
        }
    }
    /* prevent drift */
    KM = n_therm_states * n_conf_states;
    shift = new_biased_conf_energies[0];
    for(i=1; i<KM; ++i)
        shift = (shift < new_biased_conf_energies[i]) ? shift : new_biased_conf_energies[i];
    for(i=0; i<KM; ++i)
        new_biased_conf_energies[i] -= shift;
}

void _get_conf_energies(
    double *bias_energy_sequence, int *state_sequence, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_T,
    double *conf_energies)
{
    int i, K, x;
    double divisor;
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] = INFINITY;
    for( x=0; x<seq_length; ++x )
    {
        i = state_sequence[x];
        for(K=0; K<n_therm_states; ++K)
            scratch_T[K] = log_R_K_i[K * n_conf_states + i] - bias_energy_sequence[K * seq_length + x];
        divisor = _logsumexp(scratch_T, n_therm_states);
        conf_energies[i] = -_logsumexp_pair(-conf_energies[i], -divisor);
    }
}

void _get_therm_energies(
    double *biased_conf_energies, int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies)
{
    int i, K;
    for(K=0; K<n_therm_states; ++K)
    {
        for(i=0; i<n_conf_states; ++i)
            scratch_M[i] = -biased_conf_energies[K * n_conf_states + i];
        therm_energies[K] = -_logsumexp(scratch_M, n_conf_states);
    }
}

void _normalize(
    double *conf_energies, double *biased_conf_energies, double *therm_energies,
    int n_therm_states, int n_conf_states, double *scratch_M)
{
    int i, KM = n_therm_states * n_conf_states;
    double f0;
    for(i=0; i<n_conf_states; ++i)
        scratch_M[i] = -conf_energies[i];
    f0 = -_logsumexp(scratch_M, n_conf_states);
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] -= f0;
    for(i=0; i<KM; ++i)
        biased_conf_energies[i] -= f0;
    for(i=0; i<n_therm_states; ++i)
        therm_energies[i] -= f0;
}

void _estimate_transition_matrix(
    double *log_lagrangian_mult, double *conf_energies, int *count_matrix,
    int n_conf_states, double *transition_matrix)
{
    int i, j;
    int ij, ji;
    int C;
    double divisor, sum;
    for(i=0; i<n_conf_states; ++i)
    {
        sum = 0.0;
        for(j=0; j<n_conf_states; ++j)
        {
            ij = i*n_conf_states + j;
            transition_matrix[ij] = 0.0;
            /* special case: diagonal element */
            if(i == j) continue;
            ji = j*n_conf_states + i;
            C = count_matrix[ij] + count_matrix[ji];
            /* special case: this element is zero */
            if(0 == C) continue;
            /* regular case */
            divisor = _logsumexp_pair(
                log_lagrangian_mult[j] - conf_energies[i],
                log_lagrangian_mult[i] - conf_energies[j]);
            transition_matrix[ij] = C * exp(-(conf_energies[j] + divisor));
            sum += transition_matrix[ij];
        }
        /* empty row */
        if(0.0 == sum)
            transition_matrix[i*n_conf_states + i] = 1.0;
        /* too  large row */
        else if(1.0 < sum)
        {
            if(0 < count_matrix[i*n_conf_states + i])
                printf("# WARNING! THERMOTOOLS::TRAM::ESTIMATE_TRANSITION_MATRIX: T[%d,%d]=0 but C[%d,%d]=%d\n", i, i, i, i, count_matrix[i*n_conf_states + i]);
            for(j=0; j<n_conf_states; ++j)
                transition_matrix[i*n_conf_states + j] /= sum;
        }
        /* regular row */
        else
            transition_matrix[i*n_conf_states + i] = 1.0 - sum;
    }
}
