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

void _init_lagrangian_mult(int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult);

void _update_lagrangian_mult(
    double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices,
    int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult);

void _update_biased_conf_energies(
    double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices, double *bias_energy_sequence,
    int *state_sequence, int *state_counts, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_T,
    double *new_biased_conf_energies);

void _get_conf_energies(
    double *bias_energy_sequence, int *state_sequence, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_T,
    double *conf_energies);

void _get_therm_energies(
    double *biased_conf_energies, int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies);

void _normalize(
    double *conf_energies, double *biased_conf_energies, double *therm_energies,
    int n_therm_states, int n_conf_states, double *scratch_M);

void _estimate_transition_matrix(
    double *log_lagrangian_mult, double *conf_energies, int *count_matrix,
    int n_conf_states, double *scratch_M, double *transition_matrix);

#endif
