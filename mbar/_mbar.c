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

#include "../util/_util.h"

extern void _mbar_update_therm_energies(
    double *log_therm_state_counts, double *therm_energies, double *bias_energy_sequence,
    int n_therm_states, int seq_length, double *scratch_T, double *new_therm_energies)
{
    int K, x, L;
    double divisor;
    /* assume that new_therm_energies were set to INF by the caller on the first call */
    for(x=0; x<seq_length; ++x)
    {
        for(L=0; L<n_therm_states; ++L)
            scratch_T[L] = log_therm_state_counts[L] + therm_energies[L] - bias_energy_sequence[x * n_therm_states + L];
        divisor = _logsumexp_sort_kahan_inplace(scratch_T, n_therm_states);
        for(K=0; K<n_therm_states; ++K)
            new_therm_energies[K] = -_logsumexp_pair(-new_therm_energies[K], -(bias_energy_sequence[x * n_therm_states + K] + divisor));
    }
}

extern void _mbar_get_conf_energies(
    double *log_therm_state_counts, double *therm_energies,
    double *bias_energy_sequence, int *conf_state_sequence,
    int n_therm_states, int n_conf_states, int seq_length,
    double *scratch_T, double *conf_energies, double *biased_conf_energies)
{
    int i, x, L, K;
    double divisor;
    /* assume that conf_energies and biased_conf_energies were set to INF by the caller on the first call */
    for(x=0; x<seq_length; ++x)
    {
        for(L=0; L<n_therm_states; ++L)
            scratch_T[L] = log_therm_state_counts[L] + therm_energies[L] - bias_energy_sequence[x * n_therm_states + L];
        i = conf_state_sequence[x];
        if(i < 0) continue;
        divisor = _logsumexp_sort_kahan_inplace(scratch_T, n_therm_states);
        conf_energies[i] = -_logsumexp_pair(-conf_energies[i], -divisor);
        for(K=0; K<n_therm_states; ++K)
            biased_conf_energies[K * n_conf_states + i] = -_logsumexp_pair(
                -biased_conf_energies[K * n_conf_states + i],
                -(bias_energy_sequence[x * n_therm_states + K] + divisor));
    }
}

extern void _mbar_normalize(
    int n_therm_states, int n_conf_states, double *scratch_M,
    double *therm_energies, double *conf_energies, double *biased_conf_energies)
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

void _mbar_get_pointwise_unbiased_free_energies(
    int k, double *log_therm_state_counts, double *therm_energies,
    double *bias_energy_sequence,
    int n_therm_states,  int seq_length,
    double *scratch_T, double *pointwise_unbiased_free_energies)
{
    int L, x;
    double log_divisor;

    for(x=0; x<seq_length; ++x)
    {
        for(L=0; L<n_therm_states; ++L)
            scratch_T[L] = log_therm_state_counts[L] + therm_energies[L] - bias_energy_sequence[x * n_therm_states + L];
        log_divisor = _logsumexp_sort_kahan_inplace(scratch_T, n_therm_states);
        if(k==-1)
            pointwise_unbiased_free_energies[x] = log_divisor;
        else
            pointwise_unbiased_free_energies[x] = bias_energy_sequence[x * n_therm_states + k] + log_divisor - therm_energies[k];
    }
}
