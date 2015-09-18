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

#ifndef THERMOTOOLS_DTRAM
#define THERMOTOOLS_DTRAM

#define THERMOTOOLS_DTRAM_PRIOR 1.0E-10
#define THERMOTOOLS_DTRAM_LOG_PRIOR -23.025850929940457

extern void _set_lognu(
    double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states);

extern void _iterate_lognu(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij,
    int n_therm_states, int n_markov_states, double *scratch_M, double *new_log_nu_K_i);

extern void _iterate_fi(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
    int n_markov_states, double *scratch_TM, double *scratch_M, double *new_f_i);

extern void _get_p(
    double *log_nu_i, double *b_i, double *f_i, int *C_ij,
    int n_markov_states, double *scratch_M, double *p_ij);

extern void _get_fk(
    double *b_K_i, double *f_i, int n_therm_states, int n_markov_states,
    double *scratch_M, double *f_K);

#endif
