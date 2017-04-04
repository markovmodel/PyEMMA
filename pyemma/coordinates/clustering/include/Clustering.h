//
// Created by marscher on 4/3/17.
//

#ifndef PYEMMA_CLUSTERING_H
#define PYEMMA_CLUSTERING_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "metric.h"

namespace py = pybind11;

template <typename dtype>
class ClusteringBase {
public:
    ClusteringBase(const std::string& metric_s, size_t input_dimension) {
        if (metric_s == "euclidean") {
            metric = new metric::euclidean_metric<dtype>(input_dimension);
        } else if(metric_s == "minRMSD") {
            metric = new metric::min_rmsd_metric<>(input_dimension);
        } else {
            throw std::runtime_error("metric is not of {'euclidean', 'minRMSD'}");
        }
    }

    ~ClusteringBase() { delete metric; }
    py::array_t<unsigned int> assign(...) {}
    metric::metric<dtype>* metric;
};

template <typename dtype>
class KMeans : public ClusteringBase<dtype> {
public:
    KMeans(int k, const std::string& metric, size_t input_dimension) :
            ClusteringBase<dtype>(metric, input_dimension), k(k) {

    }
    py::list cluster(py::array_t<dtype, py::array::c_style> np_chunk,
                     py::list py_centers) {

    }

    void costFunction(py::array_t<dtype> np_data, py::list np_centers) {
        int i, r;
        dtype value, d;
        dtype *data, *centers;
        size_t dim, n_frames;

        value = 0.0;
        n_frames = np_data.shape(0);
        dim = np_data.shape(1);

        for (r = 0; r < np_centers.size(); r++) {
            // this is a list of numpy arrays.
            centers = (dtype *) PyArray_DATA(np_centers[r].ptr());
            for (i = 0; i < n_frames; i++) {
                value += metric->compute(&data[i * dim], &centers[0]);
            }
        }
    }

    py::array_t<dtype, py::array::c_style>
    initCentersKMpp(py::array_t<dtype, py::array::c_style|py::array::forcecast> np_data, int k, bool use_random_seed) {

    };


    void set_callback(py::function callback) { this->callback = callback; }
protected:
    int k;
    py::function callback;

    // TODO: use a unique_ptr or something, anyhow it should be a ptr for polymorphism?


};

#endif //PYEMMA_CLUSTERING_H
