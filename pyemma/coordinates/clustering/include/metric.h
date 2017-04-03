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
class metric {

public:
    metric(std::size_t dim) : dim(dim) {}

    // overload operator()?
    //template<typename ... Params>
    dtype compute() {}

    size_t dim;
};


template<class dtype>
class euclidean_metric : public metric<dtype> {
public:
    euclidean_metric(size_t dim) : metric<dtype>(dim) {}

    template<typename ... Params>
    dtype compute(dtype *a, dtype *b) {
        dtype sum = 0.0;
        for (size_t i = 0; i < metric<dtype>::dim; ++i) {
            sum += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return std::sqrt(sum);
    }

};


class min_rmsd_metric : metric<float> {
public:

    min_rmsd_metric(std::size_t dim, float *precalc_trace_centers = nullptr)
            : metric<float>(dim), buffer_a(dim), buffer_b(dim) {
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

    float *precenter_centers(float *original_centers, size_t N_centers);
};

}
#endif //PYEMMA_METRIC_H
