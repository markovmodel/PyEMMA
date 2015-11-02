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

#include "../util/_util.h"
#include "_dtram.h"

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

/***************************************************************************************************
*   secondary implementation
***************************************************************************************************/

static double _mirrored_sigmoid_term_mk2(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies,
    int Ki, int Kj, int i, int j)
/* use normally for Lagrangian multiplier update and switch i <-> j for conf energies update */
{
    double loc = log_lagrangian_mult[Kj];
    double tmp = loc;
    double err = 0.0;
    double sum = tmp;
    _kahan_summation_step(-log_lagrangian_mult[Ki], &sum, &err, &loc, &tmp);
    _kahan_summation_step(log_lagrangian_mult[Kj], &sum, &err, &loc, &tmp);
    _kahan_summation_step(-log_lagrangian_mult[Ki], &sum, &err, &loc, &tmp);
    _kahan_summation_step(conf_energies[j], &sum, &err, &loc, &tmp);
    _kahan_summation_step(-conf_energies[i], &sum, &err, &loc, &tmp);
    return _mirrored_sigmoid(sum);
}

extern void _update_lagrangian_mult_mk2(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrices,
    int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult)
{
    int i, j, K, o;
    int MM = n_conf_states * n_conf_states, Ki, Kj, KMM;
    int CK, CKij;
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K * MM;
        for(i=0; i<n_conf_states; ++i)
        {
            Ki = K * n_conf_states + i;
            /* sparsity: zero Lagrangian multipliers stay zero */
            if(-INFINITY == log_lagrangian_mult[Ki])
            {
                new_log_lagrangian_mult[Ki] = -INFINITY;
                continue;
            }
            o = 0;
            for(j=0; j<n_conf_states; ++j)
            {
                CKij = count_matrices[KMM + i*n_conf_states + j];
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    if(0 < CKij)
                        scratch_M[o++] = (double) CKij;
                    continue;
                }
                CK = CKij + count_matrices[KMM + j*n_conf_states + i];;
                Kj = K*n_conf_states+j;
                /* special case: no contribution from zero count situations */
                if(0 == CK) continue;
                /* regular case */
                scratch_M[o++] = (double) CK * _mirrored_sigmoid_term_mk2(
                    log_lagrangian_mult, bias_energies, conf_energies, Ki, Kj, i, j);
            }
            _mixed_sort(scratch_M, 0, o - 1);
            new_log_lagrangian_mult[Ki] = log(_kahan_summation(scratch_M, o));
        }
    }
}

extern void _update_conf_energies_mk2(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrices,
    int n_therm_states, int n_conf_states, double *scratch_TM, double *new_conf_energies)
{
    int i, j, K, o;
    int MM = n_conf_states * n_conf_states, Ki, Kj, KMM;
    int CK, CKij, CKji, Ci;
    for(i=0; i<n_conf_states; ++i)
    {
        /* sparsity: zero stationary weights stay zero */
        if(INFINITY == conf_energies[i])
        {
            new_conf_energies[i] = INFINITY;
            continue;
        }
        Ci = 0;
        o = 0;
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K * n_conf_states + i;
            KMM = K * MM;
            for(j=0; j<n_conf_states; ++j)
            {
                Kj = K * n_conf_states + j;
                CKij = count_matrices[KMM + i * n_conf_states + j];
                CKji = count_matrices[KMM + j * n_conf_states + i];
                /* add counts to Ci */
                Ci += CKji;
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    if(0 < CKij)
                        scratch_TM[o++] = (double) CKij;
                    continue;
                }
                CK = CKij + CKji;
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                scratch_TM[o++] = (double) CK * _mirrored_sigmoid_term_mk2(
                    log_lagrangian_mult, bias_energies, conf_energies, Kj, Ki, j, i);
            }
        }
        /* patch Ci and the total divisor together */
        if(0 < Ci)
        {
            _mixed_sort(scratch_TM, 0, o - 1);
            new_conf_energies[i] = conf_energies[i] - log((double) Ci / _kahan_summation(scratch_TM, o));  
        }
        else
            new_conf_energies[i] = INFINITY;
    }
}

extern void _estimate_transition_matrix_mk2(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrix,
    int n_conf_states, double *scratch_M, double *transition_matrix)
{
    int i, j, o;
    int ij, ji;
    int C;
    double divisor;
    for(i=0; i<n_conf_states; ++i)
    {
        if(-INFINITY == log_lagrangian_mult[i])
        {
            for(j=0; j<n_conf_states; ++j)
                transition_matrix[i * n_conf_states + j] = 0.0;
            transition_matrix[i * n_conf_states + i] = 1.0;
            continue;
        }
        divisor = exp(log_lagrangian_mult[i]);
        o = 0;
        for(j=0; j<n_conf_states; ++j)
        {
            ij = i * n_conf_states + j;
            transition_matrix[ij] = 0.0;
            /* special case: diagonal element */
            if(i == j)
            {
                if(0 < count_matrix[ij])
                    transition_matrix[ij] = (double) count_matrix[ij] / divisor;
                continue;
            }
            ji = j * n_conf_states + i;
            C = count_matrix[ij] + count_matrix[ji];
            /* special case: this element is zero */
            if(0 == C) continue;
            /* regular case */
            ++o;
            transition_matrix[ij] = (double) C * _mirrored_sigmoid_term_mk2(
                log_lagrangian_mult, bias_energies, conf_energies, i, j, i, j) / divisor;
        }
        if(0 == o) transition_matrix[i * n_conf_states + i] = 1.0;
    }
}

/***************************************************************************************************
*   primary implementation
***************************************************************************************************/

extern void _init_lagrangian_mult(
    int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult)
{
    int i, j, K;
    int MM=n_conf_states*n_conf_states, KMM;
    double sum;
    for(K=0; K<n_therm_states; ++K)
    {
        KMM = K*MM;
        for(i=0; i<n_conf_states; ++i)
        {
            sum = 0;
            for(j=0; j<n_conf_states; ++j)
                /*sum += 0.5 * (count_matrices[KMM + i*n_conf_states + j] + count_matrices[KMM + j*n_conf_states + i]);*/
                sum += count_matrices[KMM + j*n_conf_states + i];
            /*log_lagrangian_mult[K*n_conf_states + i] = log(THERMOTOOLS_DTRAM_PRIOR + sum);*/
            log_lagrangian_mult[K*n_conf_states + i] = log(sum);
        }
    }
}

extern void _update_lagrangian_mult(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrices,
    int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult)
{
    int i, j, K, o;
    int MM=n_conf_states*n_conf_states, Ki, Kj;
    int CK, CKij, CKji;
    double divisor;
    for(i=0; i<n_conf_states; ++i)
    {
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states+i;
            o = 0;
            for(j=0; j<n_conf_states; ++j)
            {
                CKij = count_matrices[K*MM + i*n_conf_states + j];
                CKji = count_matrices[K*MM + j*n_conf_states + i];
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_M[o++] = (0 == CKij) ?
                        THERMOTOOLS_DTRAM_LOG_PRIOR : log(THERMOTOOLS_DTRAM_PRIOR + (double) CKij);
                    continue;
                }
                CK = CKij + CKji;
                Kj = K*n_conf_states+j;
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                divisor = _logsumexp_pair(
                        log_lagrangian_mult[Kj] - conf_energies[i] - bias_energies[Ki],
                        log_lagrangian_mult[Ki] - conf_energies[j] - bias_energies[Kj]);
                scratch_M[o++] = log((double) CK) - bias_energies[Kj] - conf_energies[j] + log_lagrangian_mult[Ki] - divisor;
            }
            new_log_lagrangian_mult[Ki] = _logsumexp_sort_kahan_inplace(scratch_M, o);
        }
    }
}

extern void _update_conf_energies(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrices, int n_therm_states,
    int n_conf_states, double *scratch_TM, double *new_conf_energies)
{
    int i, j, K, o;
    int MM=n_conf_states*n_conf_states, Ki, Kj;
    int CK, CKij, CKji, Ci;
    double divisor, shift;
    for(i=0; i<n_conf_states; ++i)
    {
        Ci = 0;
        o = 0;
        for(K=0; K<n_therm_states; ++K)
        {
            Ki = K*n_conf_states + i;
            for(j=0; j<n_conf_states; ++j)
            {
                Kj = K*n_conf_states + j;
                CKij = count_matrices[K*MM + i*n_conf_states + j];
                CKji = count_matrices[K*MM + j*n_conf_states + i];
                /* add counts to Ci */
                Ci += CKji;
                /* special case: most variables cancel out, here */
                if(i == j)
                {
                    scratch_TM[o] = (0 == CKij) ?
                        THERMOTOOLS_DTRAM_LOG_PRIOR : log(THERMOTOOLS_DTRAM_PRIOR + (double) CKij);
                    scratch_TM[o++] += conf_energies[i];
                    continue;
                }
                CK = CKij + CKji;
                /* special case */
                if(0 == CK) continue;
                /* regular case */
                divisor = _logsumexp_pair(
                        log_lagrangian_mult[Kj] - conf_energies[i] - bias_energies[Ki],
                        log_lagrangian_mult[Ki] - conf_energies[j] - bias_energies[Kj]);
                scratch_TM[o++] = log((double) CK) - bias_energies[Ki] + log_lagrangian_mult[Kj] - divisor;
            }
        }
        /* patch Ci and the total divisor together */
        new_conf_energies[i] = _logsumexp_sort_kahan_inplace(scratch_TM, o) - log(
            n_therm_states*THERMOTOOLS_DTRAM_PRIOR + (double) Ci);
    }
    shift = new_conf_energies[0];
    for(i=1; i<n_conf_states; ++i)
        shift = (shift < new_conf_energies[i]) ? shift : new_conf_energies[i];
    for(i=0; i<n_conf_states; ++i)
        new_conf_energies[i] -= shift;
}

extern void _estimate_transition_matrix(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrix,
    int n_conf_states, double *scratch_M, double *transition_matrix)
{
    int i, j, o;
    int ij, ji;
    int C;
    double divisor, sum;
    for(i=0; i<n_conf_states; ++i)
    {
        o = 0;
        for(j=0; j<n_conf_states; ++j)
        {
            ij = i*n_conf_states + j;
            transition_matrix[ij] = 0.0;
            /* special case: diagonal element */
            if(i == j)
            {
                scratch_M[o] = (0 == count_matrix[ij]) ?
                    THERMOTOOLS_DTRAM_LOG_PRIOR : log(THERMOTOOLS_DTRAM_PRIOR + (double) count_matrix[ij]);
                scratch_M[o] -= log_lagrangian_mult[i];
                transition_matrix[ij] = exp(scratch_M[o++]);
                continue;
            }
            ji = j*n_conf_states + i;
            C = count_matrix[ij] + count_matrix[ji];
            /* special case: this element is zero */
            if(0 == C) continue;
            /* regular case */
            divisor = _logsumexp_pair(
                    log_lagrangian_mult[j] - conf_energies[i] - bias_energies[i],
                    log_lagrangian_mult[i] - conf_energies[j] - bias_energies[j]);
            scratch_M[o] =  log((double) C) - conf_energies[j] - bias_energies[j] - divisor;
            transition_matrix[ij] = exp(scratch_M[o++]);
        }
        /* compute the diagonal elements from the other elements in this line */
        sum = exp(_logsumexp_sort_kahan_inplace(scratch_M, o));
        if(0.0 == sum)
        {
            for(j=0; j<n_conf_states; ++j)
                transition_matrix[i*n_conf_states + j] = 0.0;
            transition_matrix[i*n_conf_states + i] = 1.0;
        }
        else if(1.0 != sum)
        {
            for(j=0; j<n_conf_states; ++j)
                transition_matrix[i*n_conf_states + j] /= sum;
        }
    }
}

extern void _get_therm_energies(
    double *bias_energies, double *conf_energies, int n_therm_states, int n_conf_states,
    double *scratch_M, double *therm_energies)
{
    int K, i;
    for(K=0; K<n_therm_states; ++K)
    {
        for(i=0; i<n_conf_states; ++i)
            scratch_M[i] = -(bias_energies[K*n_conf_states + i] + conf_energies[i]);
        therm_energies[K] = -_logsumexp_sort_kahan_inplace(scratch_M, n_conf_states);
    }
}

extern void _normalize(
    int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies, double *conf_energies)
{
    int K, i;
    double f0;
    for(i=0; i<n_conf_states; ++i)
        scratch_M[i] = -conf_energies[i];
    f0 = -_logsumexp_sort_kahan_inplace(scratch_M, n_conf_states);
    for(K=0; K<n_therm_states; ++K)
        therm_energies[K] -= f0;
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] -= f0;
}

extern double _get_loglikelihood(
    int *count_matrices, double *transition_matrices,
    int n_therm_states, int n_conf_states)
{
    int i, n = n_therm_states * n_conf_states * n_conf_states;
    double sum = 0.0;
    for(i=0; i<n; ++i)
    {
        if(count_matrices[i] > 0)
            sum += count_matrices[i] * log(transition_matrices[i]);
    }
    return sum;
}
