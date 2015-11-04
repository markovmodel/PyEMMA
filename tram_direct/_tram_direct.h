#ifndef TRAM_DIRECT_H
#define TRAM_DIRECT_H

void _update_lagrangian_mult(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int* state_counts,
    int n_therm_states, int n_conf_states, int iteration, double *new_lagrangian_mult);
void _update_biased_conf_weights(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, double *bias_sequence,
    int *state_sequence, int *state_counts, int *indices, int indices_length, int seq_length, double *R_K_i,
    int n_therm_states, int n_conf_states, double *scratch_TM, double *new_biased_conf_weights);
void _dtram_like_update(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int *state_counts, 
    int n_therm_states, int n_conf_states, double *scratch_M, int *scratch_M_int, double *new_biased_conf_weights);

#endif
