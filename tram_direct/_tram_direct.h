#ifndef THERMOTOOLS_TRAM_DIRECT_H
#define THERMOTOOLS_TRAM_DIRECT_H

void _tram_direct_update_lagrangian_mult(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int* state_counts,
    int n_therm_states, int n_conf_states, double *new_lagrangian_mult);

void _tram_direct_get_Ref_K_i(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices,
    int *state_counts, int n_therm_states, int n_conf_states, double *R_K_i
#ifdef TRAMMBAR
    ,
    double *therm_weights, int *equilibrium_therm_state_counts,
    double overcounting_factor
#endif
    );

void _tram_direct_update_biased_conf_weights(
    double *bias_sequence, int *state_sequence, int seq_length, double *R_K_i,
    int n_therm_states, int n_conf_states, double *new_biased_conf_weights);

void _tram_direct_dtram_like_update(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int *state_counts, 
    int n_therm_states, int n_conf_states, double *scratch_M, int *scratch_M_int, double *new_biased_conf_weights);

#endif
