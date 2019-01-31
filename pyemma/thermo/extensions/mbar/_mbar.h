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

#ifndef THERMOTOOLS_MBAR
#define THERMOTOOLS_MBAR

void _mbar_update_therm_energies(
    double *log_therm_state_counts, double *therm_energies, double *bias_energy_sequence,
    int n_therm_states, int seq_length, double *scratch_T, double *new_therm_energies);

void _mbar_get_conf_energies(
    double *log_therm_state_counts, double *therm_energies,
    double *bias_energy_sequence, int * conf_state_sequence,
    int n_therm_states, int n_conf_states, int seq_length,
    double *scratch_T, double *conf_energies, double *biased_conf_energies);

void _mbar_normalize(
    int n_therm_states, int n_conf_states, double *scratch_M,
    double *therm_energies, double *conf_energies, double *biased_conf_energies);

void _mbar_get_pointwise_unbiased_free_energies(
    int k, double *log_therm_state_counts, double *therm_energies,
    double *bias_energy_sequence,
    int n_therm_states,  int seq_length,
    double *scratch_T, double *pointwise_unbiased_free_energies);


#endif
