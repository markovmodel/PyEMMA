/*
*   Copyright 2015 Christoph Wehmeyer
*/

#ifndef REWCORE_DTRAM
#define REWCORE_DTRAM

#define REWCORE_DTRAM_PRIOR 1.0E-10
#define REWCORE_DTRAM_LOG_PRIOR -23.025850929940457

extern void rc_dtram_set_lognu(
    double *log_nu_K_i, int *C_K_ij, int n_therm_states, int n_markov_states);

extern void rc_dtram_lognu(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij,
    int n_therm_states, int n_markov_states, double *scratch_M, double *new_log_nu_K_i);

extern void rc_dtram_fi(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
    int n_markov_states, double *scratch_TM, double *scratch_M, double *new_f_i);

extern void rc_dtram_pk(
    double *log_nu_K_i, double *b_K_i, double *f_i, int *C_K_ij, int n_therm_states,
    int n_markov_states, double *scratch_M, double *p_K_ij);

extern void rc_dtram_p(
    double *log_nu_i, double *b_i, double *f_i, int *C_ij,
    int n_markov_states, double *scratch_M, double *p_ij);

extern void rc_dtram_fk(
    double *b_K_i, double *f_i, int n_therm_states, int n_markov_states,
    double *scratch_M, double *f_K);

#endif
