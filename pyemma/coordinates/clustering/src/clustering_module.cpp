//
// Created by marscher on 7/17/17.
//

#include "metric.h"
#include "regspace.h"
#include "kmeans.h"

class MaximumMetric : public Metric {
public:

    double compute_squared_d(const double* xs, const double* ys, std::size_t dim) const override {
        return _compute(xs, ys, dim);
    }
    float compute_squared_f(const float* xs, const float* ys, std::size_t dim) const override {
        return _compute(xs, ys, dim);
    }
private:
    template<typename T>
    T _compute(const T* xs, const T* ys, std::size_t dim) const {
        T result = 0.0;
        for (size_t i = 0; i < dim; ++i) {
            auto d = std::abs(xs[i] - ys[i]);
            if (d > result) {
                result = d;
            }
        }
        return result*result;
    }
};

class RMSDMetric : public Metric {
public:

    double compute_squared_d(const double* xs, const double* ys, std::size_t dim) const override {
        std::vector<float> xsCast (xs, xs+dim), ysCast (ys, ys+dim);
        return _compute(xsCast.data(), ysCast.data(), dim);
    }
    float compute_squared_f(const float* xs, const float* ys, std::size_t dim) const override {
        return _compute(xs, ys, dim);
    }

private:
    template<typename T>
    T _compute(const T* xs, const T* ys, std::size_t dim) const {
        if (dim % 3 != 0) {
            throw std::range_error("RMSDMetric is only implemented for input data with a dimension dividable by 3.");
        }
        float trace_a, trace_b;
        auto dim3 = static_cast<const int>(dim / 3);
        std::vector<float> buffer_b (ys, ys + dim);
        std::vector<float> buffer_a (xs, xs + dim);
        inplace_center_and_trace_atom_major(buffer_a.data(), &trace_a, 1, dim3);
        inplace_center_and_trace_atom_major(buffer_b.data(), &trace_b, 1, dim3);


        float msd = msd_atom_major(dim3, dim3, xs,
                                   buffer_b.data(), trace_a, trace_b, 0, nullptr);
        return msd;
    }
};

using dtype = float;

PYBIND11_MODULE(_ext, m) {
    m.doc() = "module containing clustering algorithms.";

    auto regspace_mod = m.def_submodule("regspace");
    auto kmeans_mod = m.def_submodule("kmeans");

    typedef ClusteringBase<dtype> cbase_f;
    typedef RegularSpaceClustering<dtype> regspace_f;

    // register base class first.
    py::class_<cbase_f>(m, "ClusteringBase_f")
            .def(py::init<const std::string&, std::size_t>())
            .def("assign", &cbase_f::assign_chunk_to_centers)
            .def("precenter_centers", &cbase_f::precenter_centers);
    // regular space clustering.
    py::class_<regspace_f, cbase_f>(regspace_mod, "Regspace_f")
            .def(py::init<dtype, std::size_t, const std::string&, size_t>())
            .def("cluster", &regspace_f::cluster);
    py::register_exception<MaxCentersReachedException>(regspace_mod, "MaxCentersReachedException");
    // kmeans
    typedef KMeans<dtype> kmeans_f;
    py::class_<kmeans_f, cbase_f>(kmeans_mod, "Kmeans_f")
            .def(py::init<unsigned int, const std::string&, std::size_t>(),
                 py::arg("k"), py::arg("metric"), py::arg("dim"))
             // py::arg("callback") = py::none()
            .def("cluster", &kmeans_f::cluster)
            .def("cluster_loop", &kmeans_f::cluster_loop)
            .def("init_centers_KMpp", &kmeans_f::initCentersKMpp)
            .def("cost_function", &kmeans_f::costFunction);
    // py::class_<Metric>(m, "Metric");
    py::object baseMetric = (py::object) py::module_::import("deeptime.clustering._clustering_bindings").attr("Metric");
    py::class_<RMSDMetric>(m, "RMSDMetric", baseMetric).def(py::init<>());
}
