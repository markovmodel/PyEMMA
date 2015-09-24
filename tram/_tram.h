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

#ifndef THERMOTOOLS_TRAM
#define THERMOTOOLS_TRAM

#define THERMOTOOLS_TRAM_PRIOR 1.0E-10
#define THERMOTOOLS_TRAM_LOG_PRIOR -23.025850929940457

void _set_lognu(double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states);
void _iterate_lognu(
    double *log_nu_K_i, double *f_K_i, int *C_K_ij,
    int n_therm_states, int n_markov_states, double *scratch_M, double *new_log_nu_K_i);
void _iterate_fki(
    double *log_nu_K_i, double *f_K_i, int *C_K_ij, double *b_K_x,
    int *M_x, int *N_K_i, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_markov_states, double *scratch_M, double *scratch_T,
    double *new_f_K_i);
void _get_fi(
    double *b_K_x, int *M_x, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_markov_states, double *scratch_M, double *scratch_T,
    double *f_i);
void _normalize_fki(
    double *f_i, double *f_K_i, int n_therm_states, int n_markov_states, double *scratch_M);
void _get_p(
    double *log_nu_i, double *f_i, int *C_ij,
    int n_markov_states, double *scratch_M, double *p_ij);

#endif
