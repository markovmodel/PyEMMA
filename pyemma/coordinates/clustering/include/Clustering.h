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
    ClusteringBase(const std::string& metric_s, std::size_t input_dimension) {
        if (metric_s == "euclidean") {
            metric = new metric::euclidean_metric<dtype>(input_dimension);
        } else if(metric_s == "minRMSD") {
            metric = new metric::min_rmsd_metric<float>(input_dimension);
        } else {
            throw std::runtime_error("metric is not of {'euclidean', 'minRMSD'}");
        }
    }

    ~ClusteringBase() { delete metric; }
    py::array_t<unsigned int> assign(...) {}
    metric::metric_base<dtype>* metric;
};



#endif //PYEMMA_CLUSTERING_H
