/*

    _dtram.h - dTRAM implementation in C (header file)

    author: Christoph Wehmeyer <christoph.wehmeyer@fu-berlin.de>

*/

#ifndef PYTRAM_DTRAM
#define PYTRAM_DTRAM

#include <math.h>
#include "../lse/_lse.h"

#define PYTRAM_DTRAM_PRIOR 1.0E-10
#define PYTRAM_DTRAM_LOG_PRIOR -23.025850929940457

void _log_nu_K_i_setter(
    double *log_nu_K_i,
    int *C_K_ij,
    int n_therm_states,
    int n_markov_states
);

void _log_nu_K_i_equation(
    double *log_nu_K_i,
    double *b_K_i,
    double *f_i,
    int *C_K_ij,
    int n_therm_states,
    int n_markov_states,
    double *scratch_j,
    double *new_log_nu_K_i
);

void _f_i_equation(
    double *log_nu_K_i,
    double *b_K_i,
    double *f_i,
    int *C_K_ij,
    int n_therm_states,
    int n_markov_states,
    double *scratch_K_j,
    double *scratch_j,
    double *new_f_i
);

void _p_K_ij_equation(
    double *log_nu_K_i,
    double *b_K_i,
    double *f_i,
    int *C_K_ij,
    int n_therm_states,
    int n_markov_states,
    double *scratch_j,
    double *p_K_ij
);

void _f_K_equation(
    double *b_K_i,
    double *f_i,
    int n_therm_states,
    int n_markov_states,
    double *scratch_j,
    double *f_K
);

#endif
