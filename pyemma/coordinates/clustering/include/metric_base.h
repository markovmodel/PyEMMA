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

    dtype compute(const dtype *a, const dtype *b);
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
