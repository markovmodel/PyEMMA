//
// Created by marscher on 4/3/17.
//

#ifndef PYEMMA_CLUSTERING_H
#define PYEMMA_CLUSTERING_H

#include <cstdlib>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "metric_base.h"
#include <iostream>
namespace py = pybind11;

template <typename dtype>
class ClusteringBase {
public:
    using metric_ref = std::unique_ptr<metric_base<dtype>>;
    ClusteringBase(const std::string& metric_s, std::size_t input_dimension) : input_dimension(input_dimension) {
        if (metric_s == "euclidean") {
            metric = metric_ref(new euclidean_metric<dtype>(input_dimension));
        } else if(metric_s == "minRMSD") {
            metric = metric_ref(new min_rmsd_metric<float>(input_dimension));
        } else {
            throw std::invalid_argument("metric is not of {'euclidean', 'minRMSD'}");
        }
    }

    ~ClusteringBase() = default;
    ClusteringBase(const ClusteringBase&) = delete;
    ClusteringBase&operator=(const ClusteringBase&) = delete;
    ClusteringBase(ClusteringBase&&) = default;
    ClusteringBase&operator=(ClusteringBase&&) = default;

    py::array_t<unsigned int> assign(...) {}
    metric_ref metric;
    std::size_t input_dimension;
};



#endif //PYEMMA_CLUSTERING_H
