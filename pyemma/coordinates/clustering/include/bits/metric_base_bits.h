//
// Created by marscher on 7/21/17.
//

#ifndef PYEMMA_METRIC_BASE_BITS_H
#define PYEMMA_METRIC_BASE_BITS_H

#include "../metric_base.h"

/*
 * minRMSD distance function
 * a: centers
 * b: frames
 * n: dimension of one frame
 * buffer_a: pre-allocated buffer to store a copy of centers
 * buffer_b: pre-allocated buffer to store a copy of frames
 * trace_a_precalc: pre-calculated trace to centers (pointer to one value)
 */
template <typename dtype>
inline dtype min_rmsd_metric<dtype>::compute(const dtype *a, const dtype *b) {
    float msd;
    float trace_a, trace_b;

    if (!has_trace_a_been_precalculated) {
        buffer_a.assign(a, a + parent_t::dim * sizeof(float));
        buffer_b.assign(b, b + parent_t::dim * sizeof(float));

        // assert(parent_t::dim % 3 == 0);

        inplace_center_and_trace_atom_major(buffer_b.data(), &trace_b, 1, parent_t::dim / 3);

    } else {
        // only copy b, since a has been pre-centered,
        buffer_b.assign(b, b + parent_t::dim * sizeof(float));

        inplace_center_and_trace_atom_major(buffer_b.data(), &trace_b, 1, parent_t::dim / 3);
        trace_a = *trace_centers.data();
    }

    msd = msd_atom_major(parent_t::dim / 3, parent_t::dim / 3, a, buffer_b.data(), trace_a, trace_b, 0, NULL);
    return std::sqrt(msd);
}

template<typename dtype>
inline float * min_rmsd_metric<dtype>::precenter_centers(float *original_centers, std::size_t N_centers) {
    centers_precentered.reserve(N_centers*parent_t::dim);
    centers_precentered.assign(original_centers, original_centers + (N_centers * parent_t::dim));
    trace_centers.reserve(N_centers);
    float *trace_centers_p = trace_centers.data();

    /* Parallelize centering of cluster generators */
    /* Note that this is already OpenMP-enabled */
    for (int j = 0; j < N_centers; ++j) {
        inplace_center_and_trace_atom_major(&centers_precentered[j * parent_t::dim],
                                            &trace_centers_p[j], 1, parent_t::dim / 3);
    }
    return centers_precentered.data();
}


#endif //PYEMMA_METRIC_BASE_BITS_H
