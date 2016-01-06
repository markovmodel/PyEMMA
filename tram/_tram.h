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

/* #define THERMOTOOLS_TRAM_PRIOR 1.0E-10 */
/* #define THERMOTOOLS_TRAM_LOG_PRIOR -23.025850929940457 */

#define THERMOTOOLS_TRAM_PRIOR 0
#define THERMOTOOLS_TRAM_LOG_PRIOR -INFINITY

void _tram_init_lagrangian_mult(int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult);

void _tram_update_lagrangian_mult(
    double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices, int* state_counts,
    int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult);

void _tram_update_biased_conf_energies(
    double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices, double *bias_energy_sequence,
    int *state_sequence, int *state_counts, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_T,
    double *new_biased_conf_energies);

void _tram_get_conf_energies(
    double *bias_energy_sequence, int *state_sequence, int seq_length, double *log_R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_M, double *scratch_T,
    double *conf_energies);

void _tram_get_therm_energies(
    double *biased_conf_energies, int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies);

void _tram_normalize(
    double *conf_energies, double *biased_conf_energies, double *therm_energies,
    int n_therm_states, int n_conf_states, double *scratch_M);

void _tram_estimate_transition_matrix(
    double *log_lagrangian_mult, double *conf_energies, int *count_matrix,
    int n_conf_states, double *scratch_M, double *transition_matrix);

double _tram_log_likelihood_lower_bound(
    double *old_log_lagrangian_mult, double *new_log_lagrangian_mult,
    double *old_biased_conf_energies, double *new_biased_conf_energies,
    int *count_matrices,  int *state_counts,
    int n_therm_states, int n_conf_states,
    double *bias_energy_sequence, int *state_sequence, int seq_length,
    double *scratch_T, double *scratch_M, double *scratch_TM, double *scratch_MM);

void _get_log_R_K_i(double *log_lagrangian_mult, double *biased_conf_energies, int *count_matrices,
    int *state_counts, int n_therm_states, int n_conf_states, double *scratch_M,
    double *log_R_K_i);

void _get_pointwise_unbiased_free_energies(
    int k, double *bias_energy_sequence, double *therm_energies, int *state_sequence,
    int seq_length, double *log_R_K_i, int n_therm_states, int n_conf_states,
    double *scratch_T, double *pointwise_unbiased_free_energies);

void _get_unbiased_user_free_energies(double *unbiased_pointwise_free_energies,
    int *user_index_sequence, int seq_length, int n_user_states, double *unbiased_user_free_energies);

double _get_unbiased_expectation(double *unbiased_pointwise_free_energies, double *observable_sequence, int seq_length);

#endif
