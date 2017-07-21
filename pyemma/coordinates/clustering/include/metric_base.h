//
// Created by marscher on 3/31/17.
//

#ifndef PYEMMA_METRIC_H
#define PYEMMA_METRIC_H

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <cmath>

#include <vector>

/**
 *
 * @tparam dtype eg. float, double
 */
template<typename dtype>
class metric_base {

public:
    metric_base(std::size_t dim) : dim(dim) {}
    virtual ~metric_base() = default;

    virtual dtype compute(const dtype *, const dtype *) = 0;

    size_t dim;
};

#include <iostream>
template<class dtype>
class euclidean_metric : public metric_base<dtype> {
public:
    euclidean_metric(size_t dim) : metric_base<dtype>(dim) {}
    ~euclidean_metric() = default;

    dtype compute(const dtype * const a, const dtype * const b) {
        dtype sum = 0.0;
        for (size_t i = 0; i < metric_base<dtype>::dim; ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

};

template<typename dtype>
class min_rmsd_metric : public metric_base<dtype> {

    static_assert(std::is_same<dtype, float>::value, "only implemented for floats");

public:

    min_rmsd_metric(std::size_t dim, float *precalc_trace_centers = nullptr)
            : metric_base<float>(dim), buffer_a(dim), buffer_b(dim) {
        trace_a_precalc = precalc_trace_centers != nullptr;
    }
    ~min_rmsd_metric() = default;

    // TODO: actually a generic type argument makes no sense, because rmsd in mdtraj is only impled for float...
    dtype compute(const dtype *a, const dtype *b) {}

private:
    std::vector<float> buffer_a, buffer_b;
    /**
     * only used during cluster assignment. Avoids precentering the centers in every step.
     */
    std::vector<float> centers_precentered;
    std::vector<float> trace_centers;
    bool trace_a_precalc;

    float *precenter_centers(float *original_centers, std::size_t N_centers) {
        /*
        distance = minRMSD_distance;
        centers_precentered.reserve(N_centers*dim*sizeof(float));
        trace_centers.reserve(N_centers*sizeof(float));
        dists = malloc(N_centers*sizeof(float));
        if(!centers_precentered || !dists || !trace_centers_p) {
            ret = ASSIGN_ERR_NO_MEMORY;
        }

        if (ret == ASSIGN_SUCCESS) {
            memcpy(centers_precentered, centers, N_centers*dim*sizeof(float));

            // Parallelize centering of cluster generators
            // Note that this is already OpenMP-enabled
            for (j = 0; j < N_centers; ++j) {
                inplace_center_and_trace_atom_major(&centers_precentered[j*dim],
                                                    &trace_centers_p[j], 1, dim/3);
            }
            centers = centers_precentered;
        }
    */
    }
};

#endif //PYEMMA_METRIC_H
