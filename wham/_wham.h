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

#ifndef THERMOTOOLS_WHAM
#define THERMOTOOLS_WHAM

extern void _iterate_fi(
    double *log_N_K, double *log_N_i, double *f_K, double *b_K_i,
    int n_therm_states, int n_markov_states, double *scratch_M, double *scratch_T, double *f_i);

extern void _iterate_fk(
    double *f_i, double *b_K_i, int n_therm_states, int n_markov_states,
    double *scratch_M, double *f_K);

#endif
