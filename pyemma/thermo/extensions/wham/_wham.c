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

extern void _wham_update_conf_energies(
    double *log_therm_state_counts, double *log_conf_state_counts,
    double *therm_energies, double *bias_energies,
    int n_therm_states, int n_conf_states, double *scratch_T, double *conf_energies)
{
    int i, K;
    for(i=0; i<n_conf_states; ++i)
    {
        for(K=0; K<n_therm_states; ++K)
            scratch_T[K] = log_therm_state_counts[K]
                         - bias_energies[K * n_conf_states + i]
                         + therm_energies[K];
        conf_energies[i] = _logsumexp_sort_kahan_inplace(scratch_T, n_therm_states)
                         - log_conf_state_counts[i];
    }
}

extern void _wham_update_therm_energies(
    double *conf_energies, double *bias_energies, int n_therm_states, int n_conf_states,
    double *scratch_M, double *therm_energies)
{
    int i, K, KM;
    for(K=0; K<n_therm_states; ++K)
    {
        KM = K*n_conf_states;
        for(i=0; i<n_conf_states; ++i)
            scratch_M[i] = -(bias_energies[KM + i] + conf_energies[i]);
        therm_energies[K] = -_logsumexp_sort_kahan_inplace(scratch_M, n_conf_states);
    }
}

extern void _wham_normalize(
    int n_therm_states, int n_conf_states,
    double *scratch_M, double *therm_energies, double *conf_energies)
{
    int K, i;
    double f0;
    for(i=0; i<n_conf_states; ++i)
        scratch_M[i] = -conf_energies[i];
    f0 = -_logsumexp_sort_kahan_inplace(scratch_M, n_conf_states);
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] -= f0;
    for(K=0; K<n_therm_states; ++K)
        therm_energies[K] -= f0;
}

extern double _wham_get_loglikelihood(
    int *therm_state_counts, int *conf_state_counts,
    double *therm_energies, double *conf_energies,
    int n_therm_states, int n_conf_states, double *scratch_S)
{
    int i, o = 0;
    for(i=0; i<n_therm_states; ++i)
    {
        if(therm_state_counts[i] > 0)
            scratch_S[o++] = therm_state_counts[i] * therm_energies[i];
    }
    for(i=0; i<n_conf_states; ++i)
    {
        if(conf_state_counts[i] > 0)
            scratch_S[o++] = -conf_state_counts[i] * conf_energies[i];
    }
    _mixed_sort(scratch_S, 0, o - 1);
    return _kahan_summation(scratch_S, o);
}
