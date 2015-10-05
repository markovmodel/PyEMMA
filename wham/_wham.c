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
    double *log_N_K, double *log_N_i, double *f_K, double *b_K_i,
    int n_therm_states, int n_markov_states, double *scratch_T, double *f_i)
{
    int i, K;
    double shift;
    for(i=0; i<n_markov_states; ++i)
    {
        for(K=0; K<n_therm_states; ++K)
            scratch_T[K] = log_N_K[K] - b_K_i[K*n_markov_states + i] + f_K[K];
        f_i[i] = _logsumexp(scratch_T, n_therm_states) - log_N_i[i];
    }
    shift = f_i[0];
    for(i=1; i<n_markov_states; ++i)
        shift = (shift < f_i[i]) ? shift : f_i[i];
    for(i=0; i<n_markov_states; ++i)
        f_i[i] -= shift;
}

extern void _update_therm_energies(
    double *f_i, double *b_K_i, int n_therm_states, int n_markov_states,
    double *scratch_M, double *f_K)
{
    int i, K, KM;
    for(K=0; K<n_therm_states; ++K)
    {
        KM = K*n_markov_states;
        for(i=0; i<n_markov_states; ++i)
            scratch_M[i] = -(b_K_i[KM + i] + f_i[i]);
        f_K[K] = -_logsumexp(scratch_M, n_markov_states);
    }
}

extern void _normalize(
    int n_therm_states, int n_markov_states, double *scratch_M, double *f_K, double *f_i)
{
    int K, i;
    double f0;
    for(i=0; i<n_markov_states; ++i)
        scratch_M[i] = -f_i[i];
    f0 = -_logsumexp(scratch_M, n_markov_states);
    for(i=0; i<n_markov_states; ++i)
        f_i[i] -= f0;
    for(K=0; K<n_therm_states; ++K)
        f_K[K] -= f0;
}
