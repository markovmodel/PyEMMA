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
#include "_mbar_direct.h"

void _update_therm_weights(
    int *therm_state_counts, double *therm_weights, double *bias_weight_sequence,
    int n_therm_states, int seq_length, double *scratch_T, double *new_therm_weights)
{
    int K, x, L;
    double divisor, maximum;
    for(K=0; K<n_therm_states; ++K)
        new_therm_weights[K] = 0;
    for(x=0; x<seq_length; ++x)
    {
        divisor = 0;
        for(L=0; L<n_therm_states; ++L)
            divisor += (double)therm_state_counts[L] * bias_weight_sequence[L * seq_length + x] / therm_weights[L] ;
        for(K=0; K<n_therm_states; ++K)
            new_therm_weights[K] += bias_weight_sequence[K * seq_length + x] / divisor;
    }
    maximum = 0;
    for(K=0; K<n_therm_states; ++K)
        if(new_therm_weights[K] > maximum)
            maximum = new_therm_weights[K];
    for(K=0; K<n_therm_states; ++K)
        new_therm_weights[K] /= maximum;
}
