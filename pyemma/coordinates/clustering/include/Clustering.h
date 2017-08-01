//
// Created by marscher on 4/3/17.
//

#ifndef PYEMMA_CLUSTERING_H
#define PYEMMA_CLUSTERING_H

#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "metric_base.h"

namespace py = pybind11;

template <typename dtype>
class ClusteringBase {
public:
    using np_array = py::array_t<dtype, py::array::c_style | py::array::forcecast>;

    ClusteringBase(const std::string& metric_s, std::size_t input_dimension) : input_dimension(input_dimension) {
        if (metric_s == "euclidean") {
            typedef euclidean_metric<dtype> eucl;
            metric = std::unique_ptr<eucl>( new eucl(input_dimension));
        } else if(metric_s == "minRMSD") {
            typedef min_rmsd_metric<float> min_rmsd_t;
            metric = std::unique_ptr<min_rmsd_t>(new min_rmsd_t(input_dimension));
        } else {
            throw std::invalid_argument("metric is not of {'euclidean', 'minRMSD'}");
        }
    }

    virtual ~ClusteringBase()= default;
    ClusteringBase(const ClusteringBase&) = delete;
    ClusteringBase&operator=(const ClusteringBase&) = delete;
    ClusteringBase(ClusteringBase&&) = default;
    ClusteringBase&operator=(ClusteringBase&&) = default;

    std::unique_ptr<metric_base<dtype>> metric;
    std::size_t input_dimension;

    py::array_t<int> assign_chunk_to_centers(const py::array_t<dtype, py::array::c_style>& chunk,
                                             const py::array_t<dtype, py::array::c_style>& centers,
                                             unsigned int n_threads) const {
        return metric->assign_chunk_to_centers(chunk, centers, n_threads);
    }
};



#endif //PYEMMA_CLUSTERING_H
