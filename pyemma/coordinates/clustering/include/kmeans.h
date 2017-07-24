//
// Created by marscher on 4/3/17.
//

#ifndef PYEMMA_KMEANS_H
#define PYEMMA_KMEANS_H

#include "Clustering.h"

#include <limits>
#include <ctime>

namespace py = pybind11;


template <typename dtype>
class KMeans : public ClusteringBase<dtype> {
public:
    KMeans(int k, const std::string& metric, size_t input_dimension,
           py::function& callback, const std::string& init_method) :
            ClusteringBase<dtype>(metric, input_dimension), k(k), callback(callback) {

    }
    /**
     * performs kmeans clustering on the given data chunk, provided a list of centers.
     * @param np_chunk
     * @param py_centers
     * @return updated centers.
     */
    py::list cluster(const py::array_t<dtype, py::array::c_style>& np_chunk,
                                    py::list py_centers);

    dtype costFunction(py::array_t<dtype, py::array::c_style> np_data, py::list np_centers);

    // TODO: this could be static, in order to get an initial guess for the centers.
    py::array_t<dtype, py::array::c_style>
    initCentersKMpp(py::array_t<dtype, py::array::c_style|py::array::forcecast> np_data, int k, bool use_random_seed);


    void set_callback(const py::function& callback) { this->callback = callback; }
protected:
    int k;
    py::function callback;

};

#include "bits/kmeans_bits.h"

#endif //PYEMMA_KMEANS_H
