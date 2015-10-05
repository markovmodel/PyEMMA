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
#include "_wham.h"

/* old m$ visual studio is not c99 compliant (vs2010 eg. is not) */
#ifdef _MSC_VER
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#endif

extern void _update_conf_energies(
    double *log_therm_state_counts, double *log_conf_state_counts, double *therm_energies, double *bias_energies,
    int n_therm_states, int n_conf_states, double *scratch_T, double *conf_energies)
{
    int i, K;
    double shift;
    for(i=0; i<n_conf_states; ++i)
    {
        for(K=0; K<n_therm_states; ++K)
            scratch_T[K] = log_therm_state_counts[K] - bias_energies[K*n_conf_states + i] + therm_energies[K];
        conf_energies[i] = _logsumexp(scratch_T, n_therm_states) - log_conf_state_counts[i];
    }
    shift = conf_energies[0];
    for(i=1; i<n_conf_states; ++i)
        shift = (shift < conf_energies[i]) ? shift : conf_energies[i];
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] -= shift;
}

extern void _update_therm_energies(
    double *conf_energies, double *bias_energies, int n_therm_states, int n_conf_states,
    double *scratch_M, double *therm_energies)
{
    int i, K, KM;
    for(K=0; K<n_therm_states; ++K)
    {
        KM = K*n_conf_states;
        for(i=0; i<n_conf_states; ++i)
            scratch_M[i] = -(bias_energies[KM + i] + conf_energies[i]);
        therm_energies[K] = -_logsumexp(scratch_M, n_conf_states);
    }
}

extern void _normalize(
    int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies, double *conf_energies)
{
    int K, i;
    double f0;
    for(i=0; i<n_conf_states; ++i)
        scratch_M[i] = -conf_energies[i];
    f0 = -_logsumexp(scratch_M, n_conf_states);
    for(i=0; i<n_conf_states; ++i)
        conf_energies[i] -= f0;
    for(K=0; K<n_therm_states; ++K)
        therm_energies[K] -= f0;
}
