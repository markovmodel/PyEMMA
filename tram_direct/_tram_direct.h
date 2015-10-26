#ifndef TRAM_DIRECT_H
#define TRAM_DIRECT_H

struct my_sparse {
    int *rows;
    int *cols;
    int length;
};

void _update_lagrangian_mult(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, int* state_counts,
    int n_therm_states, int n_conf_states, double *new_lagrangian_mult);
struct my_sparse _update_biased_conf_weights(
    double *lagrangian_mult, double *biased_conf_weights, int *count_matrices, double *bias_sequence,
    int *state_sequence, int *state_counts, int seq_length, double *R_K_i,
    int n_therm_states, int n_conf_states, int check_overlap, double *scratch_TM, double *new_biased_conf_weights);

#endif
