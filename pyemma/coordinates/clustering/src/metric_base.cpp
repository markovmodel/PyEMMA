//
// Created by marscher on 4/3/17.
//

#include <metric_base.h>
// for mdtrajs minRMSD impl
#include <theobald_rmsd.h>
#include <center.h>

#include <cassert>

namespace metric {
/*
 * minRMSD distance function
 * a: centers
 * b: frames
 * n: dimension of one frame
 * buffer_a: pre-allocated buffer to store a copy of centers
 * buffer_b: pre-allocated buffer to store a copy of frames
 * trace_a_precalc: pre-calculated trace to centers (pointer to one value)
 */
float min_rmsd_metric::compute(float *a, float *b) {
    float msd;
    float trace_a, trace_b;

    if (!trace_a_precalc) {
        buffer_a.assign(a, a + dim * sizeof(float));
        buffer_b.assign(b, b + dim * sizeof(float));

        assert(dim % 3 == 0);

        inplace_center_and_trace_atom_major(buffer_a.data(), &trace_a, 1, dim / 3);
        inplace_center_and_trace_atom_major(buffer_b.data(), &trace_b, 1, dim / 3);

    } else {
        // only copy b, since a has been pre-centered,
        buffer_b.assign(b, b + dim * sizeof(float));

        inplace_center_and_trace_atom_major(buffer_b.data(), &trace_b, 1, dim / 3);
        trace_a = *trace_centers.data();
    }

    msd = msd_atom_major(dim / 3, dim / 3, a, buffer_b.data(), trace_a, trace_b, 0, NULL);
    return std::sqrt(msd);
}


float *min_rmsd_metric::precenter_centers(float *original_centers, size_t N_centers) {
    centers_precentered.reserve(N_centers*dim);
    centers_precentered.assign(original_centers, original_centers + (N_centers * dim));
    trace_centers.reserve(N_centers);
    float *trace_centers_p = trace_centers.data();

    /* Parallelize centering of cluster generators */
    /* Note that this is already OpenMP-enabled */
    for (int j = 0; j < N_centers; ++j) {
        inplace_center_and_trace_atom_major(&centers_precentered[j * dim],
                                            &trace_centers_p[j], 1, dim / 3);
    }
    return centers_precentered.data();
}

}