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

namespace metric {
/**
 *
 * @tparam dtype eg. float, double
 */
template<typename dtype>
class metric_base {

public:
    metric_base(std::size_t dim) : dim(dim) {}

    // overload operator()?
    template<typename ... Params>
    dtype compute(Params...) { return dtype(); }

    size_t dim;
};


template<class dtype>
class euclidean_metric : public metric_base<dtype> {
public:
    euclidean_metric(size_t dim) : metric_base<dtype>(dim) {}

    template<typename ... Params>
    dtype compute(dtype *a, dtype *b) {
        dtype sum = 0.0;
        for (size_t i = 0; i < metric_base<dtype>::dim; ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

};

template <typename dummy>
class min_rmsd_metric : public metric_base<dummy> {
public:

    min_rmsd_metric(std::size_t dim, float *precalc_trace_centers = nullptr)
            : metric_base<float>(dim), buffer_a(dim), buffer_b(dim) {
        trace_a_precalc = precalc_trace_centers != nullptr;
    }

    // TODO: actually a generic type argument makes no sense, because rmsd in mdtraj is only impled for float...
    float compute(float *a, float *b);

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

}
#endif //PYEMMA_METRIC_H
