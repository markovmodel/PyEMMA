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

void _iterate_fk(
    double *log_N_K, double *f_K, double *b_K_x,
    int n_therm_states, int seq_length, double *scratch_T, double *new_f_K)
{
    int K, x, L;
    double divisor, shift;
    for(K=0; K<n_therm_states; ++K)
        new_f_K[K] = INFINITY;
    for(x=0; x<seq_length; ++x)
    {
        for(L=0; L<n_therm_states; ++L)
            scratch_T[L] = log_N_K[L] + f_K[L] - b_K_x[L * seq_length + x];
        divisor = _logsumexp(scratch_T, n_therm_states);
        for(K=0; K<n_therm_states; ++K)
            new_f_K[K] = -_logsumexp_pair(-new_f_K[K], -b_K_x[K * seq_length + x] - divisor);
    }
    shift = new_f_K[0];
    for(K=1; K<n_therm_states; ++K)
        shift = (shift < new_f_K[K]) ? shift : new_f_K[K];
    for(K=1; K<n_therm_states; ++K)
        new_f_K[K] -= shift;
}





