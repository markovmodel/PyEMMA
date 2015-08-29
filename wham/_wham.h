/*
*   Copyright 2015 Christoph Wehmeyer
*/

#ifndef REWCORE_WHAM
#define REWCORE_WHAM

extern void rc_wham_fi(
    double *log_N_K, double *log_N_i, double *f_K, double *b_K_i,
    int n_therm_states, int n_markov_states, double *scratch, double *f_i);

extern void rc_wham_fk(
    double *f_i, double *b_K_i, int n_therm_states, int n_markov_states,
    double *scratch, double *f_K);

extern void rc_wham_normalize(double *f_i, int n_markov_states, double *scratch);

#endif
