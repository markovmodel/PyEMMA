/*
* This file is part of thermotools.
*
* Copyright 2015-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

#ifndef THERMOTOOLS_UTIL
#define THERMOTOOLS_UTIL

/***************************************************************************************************
*   C99 compatibility for macros INFINITY and NAN
***************************************************************************************************/

#include <math.h>
#ifdef _MSC_VER
    /* handle Microsofts C99 incompatibility */
    #include <float.h>
    #define INFINITY (DBL_MAX+DBL_MAX)
    #define NAN (INFINITY-INFINITY)
#else
    /* if not available otherwise, define INFINITY/NAN in the GNU style */
    #ifndef INFINITY
        #define INFINITY (1.0/0.0)
    #endif
    #ifndef NAN
        #define NAN (INFINITY-INFINITY)
    #endif
#endif

/***************************************************************************************************
*   sorting
***************************************************************************************************/

void _mixed_sort(double *array, int L, int R);

/***************************************************************************************************
*   direct summation schemes
***************************************************************************************************/

void _kahan_summation_step(
    double new_value, double *sum, double *err, double *loc, double *tmp);
double _kahan_summation(double *array, int size);

/***************************************************************************************************
*   logspace summation schemes
***************************************************************************************************/

double _logsumexp(double *array, int size, double array_max);
double _logsumexp_kahan_inplace(double *array, int size, double array_max);
double _logsumexp_sort_inplace(double *array, int size);
double _logsumexp_sort_kahan_inplace(double *array, int size);
double _logsumexp_pair(double a, double b);

/***************************************************************************************************
*   counting states and transitions
***************************************************************************************************/

int _get_therm_state_break_points(int *T_x, int seq_length, int *break_points);

/***************************************************************************************************
*   bias calculation tools
***************************************************************************************************/

void _get_umbrella_bias(
    double *traj, double *umbrella_centers, double *force_constants,
    double *width, double *half_width,
    int nsamples, int nthermo, int ndim, double *bias);

/***************************************************************************************************
*   transition matrix renormalization
***************************************************************************************************/

void _renormalize_transition_matrix(double *p, int n_conf_states, double *scratch_M);

#endif
