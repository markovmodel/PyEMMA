/*
*   Copyright 2015 Christoph Wehmeyer
*/

#ifndef REWCORE_DTRAM
#define REWCORE_DTRAM

#define REWCORE_DTRAM_PRIOR 1.0E-10
#define REWCORE_DTRAM_LOG_PRIOR -23.025850929940457

extern void rc_log_nu_K_i_setter(
    double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states);

extern void rc_log_nu_K_i_equation(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij,
    int n_therm_states, int n_markov_states, double *scratch_j, double *new_log_nu_K_i);

extern void rc_f_i_equation(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
    int n_markov_states, double *scratch_K_j, double *scratch_j, double *new_f_i);

extern void rc_p_K_ij_equation(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
    int n_markov_states, double *scratch_j, double *p_K_ij);

extern void rc_f_K_equation(
    double *b_K_i, double *f_i, int n_therm_states, int n_markov_states,
    double *scratch_j, double *f_K);

#endif
