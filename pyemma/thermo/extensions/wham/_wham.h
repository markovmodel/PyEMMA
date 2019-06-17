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

void _wham_update_conf_energies(
    double *log_therm_state_counts, double *log_conf_state_counts,
    double *therm_energies, double *bias_energies,
    int n_therm_states, int n_conf_states, double *scratch_T, double *conf_energies);

void _wham_update_therm_energies(
    double *conf_energies, double *bias_energies, int n_therm_states, int n_conf_states,
    double *scratch_M, double *therm_energies);

void _wham_normalize(
    int n_therm_states, int n_conf_states,
    double *scratch_M, double *therm_energies, double *conf_energies);

double _wham_get_loglikelihood(
    int *therm_state_counts, int *conf_state_counts,
    double *therm_energies, double *conf_energies,
    int n_therm_states, int n_conf_states, double *scratch_S);

#endif
