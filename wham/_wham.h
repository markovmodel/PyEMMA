/*
*   Copyright 2015 Christoph Wehmeyer
*/

#ifndef THERMOTOOLS_WHAM
#define THERMOTOOLS_WHAM

extern void _wham_fi(
    double *log_N_K, double *log_N_i, double *f_K, double *b_K_i,
    int n_therm_states, int n_markov_states, double *scratch_T, double *f_i);

extern void _wham_fk(
    double *f_i, double *b_K_i, int n_therm_states, int n_markov_states,
    double *scratch_M, double *f_K);

extern void _wham_normalize(double *f_i, int n_markov_states, double *scratch_M);

#endif
