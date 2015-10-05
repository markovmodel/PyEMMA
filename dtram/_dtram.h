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

extern void _init_lagrangian_mult(
    int *count_matrices, int n_therm_states, int n_conf_states, double *log_lagrangian_mult);

extern void _update_lagrangian_mult(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrices,
    int n_therm_states, int n_conf_states, double *scratch_M, double *new_log_lagrangian_mult);

extern void _update_conf_energies(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrices, int n_therm_states,
    int n_conf_states, double *scratch_TM, double *new_conf_energies);

extern void _estimate_transition_matrix(
    double *log_lagrangian_mult, double *bias_energies, double *conf_energies, int *count_matrix,
    int n_conf_states, double *scratch_M, double *transition_matrix);

extern void _get_therm_energies(
    double *bias_energies, double *conf_energies, int n_therm_states, int n_conf_states,
    double *scratch_M, double *therm_energies);

extern void _normalize(
    int n_therm_states, int n_conf_states, double *scratch_M, double *therm_energies, double *conf_energies);

#endif
