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
#include "_mbar.h"

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

extern void _update_therm_energies(
    double *log_therm_state_counts, double *therm_energies, double *bias_energy_sequence,
    int n_therm_states, int seq_length, double *scratch_T, double *new_therm_energies)
{
    int K, x, L;
    double divisor, shift;
    for(K=0; K<n_therm_states; ++K)
        new_therm_energies[K] = INFINITY;
    for(x=0; x<seq_length; ++x)
    {
        for(L=0; L<n_therm_states; ++L)
            scratch_T[L] = log_therm_state_counts[L] + therm_energies[L] - bias_energy_sequence[L * seq_length + x];
        divisor = _logsumexp(scratch_T, n_therm_states);
        for(K=0; K<n_therm_states; ++K)
            new_therm_energies[K] = -_logsumexp_pair(-new_therm_energies[K], -bias_energy_sequence[K * seq_length + x] - divisor);
    }
    shift = new_therm_energies[0];
    for(K=1; K<n_therm_states; ++K)
        shift = (shift < new_therm_energies[K]) ? shift : new_therm_energies[K];
    for(K=0; K<n_therm_states; ++K)
        new_therm_energies[K] -= shift;
}

extern void _normalize(
    double *log_therm_state_counts, double *bias_energy_sequence,
    int n_therm_states, int seq_length,
    double *scratch_T, double *therm_energies)
{
    int K, x, L;
    double divisor, f0 = INFINITY;
    for(x=0; x<seq_length; ++x)
    {
        for(L=0; L<n_therm_states; ++L)
            scratch_T[L] = log_therm_state_counts[L] + therm_energies[L] - bias_energy_sequence[L * seq_length + x];
        divisor = _logsumexp(scratch_T, n_therm_states);
        for(K=0; K<n_therm_states; ++K)
            f0 = -_logsumexp_pair(-f0, -divisor);
    }
    for(K=0; K<n_therm_states; ++K)
        therm_energies[K] -= f0;
}

extern void _get_conf_energies(
    double *log_therm_state_counts, double *therm_energies,
    double *bias_energy_sequence, int * conf_state_sequence,
    int n_therm_states, int n_conf_states, int seq_length,
    double *scratch_M, double *scratch_T, double *conf_energies)
{
    int i, x, L;
    double f0;
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] = INFINITY;
    for(x=0; x<seq_length; ++x)
    {
        for(L=0; L<n_therm_states; ++L)
            scratch_T[L] = log_therm_state_counts[L] + therm_energies[L] - bias_energy_sequence[L * seq_length + x];
        i = conf_state_sequence[x];
        conf_energies[i] = -_logsumexp_pair(-conf_energies[i], -_logsumexp(scratch_T, n_therm_states));
    }
    for(i=0; i<n_conf_states; ++i)
        scratch_M[i] = -conf_energies[i];
    f0 = -_logsumexp(scratch_M, n_conf_states);
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] -= f0;
}
