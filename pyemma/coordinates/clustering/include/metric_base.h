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

#include <theobald_rmsd.h>
#include <center.h>

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
    using parent_t = metric_base<dtype>;
    min_rmsd_metric(std::size_t dim, float *precalc_trace_centers = nullptr)
            : metric_base<float>(dim), buffer_a(dim), buffer_b(dim) {
        has_trace_a_been_precalculated = precalc_trace_centers != nullptr;
    }
    ~min_rmsd_metric() = default;

    dtype compute(const dtype *a, const dtype *b);/* {
        float msd;
        float trace_a, trace_b;
        size_t n = parent_t::dim;
        float* buffer_a_ptr = buffer_a.data();
        float* buffer_b_ptr = buffer_b.data();

        if (! has_trace_a_been_precalculated) {
            memcpy(buffer_a_ptr, a, n*sizeof(float));
            memcpy(buffer_b_ptr, b, n*sizeof(float));

            inplace_center_and_trace_atom_major(buffer_a_ptr, &trace_a, 1, n/3);
            inplace_center_and_trace_atom_major(buffer_b_ptr, &trace_b, 1, n/3);
        } else {
            // only copy b, since a has been pre-centered,
            memcpy(buffer_b_ptr, b, n*sizeof(float));
            inplace_center_and_trace_atom_major(buffer_b_ptr, &trace_b, 1, n/3);
            trace_a = trace_a_precentered;
        }

        msd = msd_atom_major(n/3, n/3, a, buffer_b_ptr, trace_a, trace_b, 0, NULL);
        return std::sqrt(msd);
    }*/
    /**
     * TODO: this can only be used during assignment?! so it should be moved to ClusteringBase::assign
     * @param original_centers
     * @param N_centers
     * @return
     */
    float *precenter_centers(float *original_centers, std::size_t N_centers);

private:
    std::vector<float> buffer_a, buffer_b;
    /**
     * only used during cluster assignment. Avoids precentering the centers in every step.
     */
    std::vector<float> centers_precentered;
    std::vector<float> trace_centers;
    bool has_trace_a_been_precalculated;
    float trace_a_precentered;
};

#include "bits/metric_base_bits.h"

#endif //PYEMMA_METRIC_H
