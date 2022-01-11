//
// Created by marscher on 7/17/17.
//

#include <center.h>
#include <theobald_rmsd.h>
#include "register_clustering.h"

struct RMSDMetric {
    template<typename T>
    static T compute_squared(const T* xs, const T* ys, std::size_t dim) {
        if (dim % 3 != 0) {
            throw std::range_error("RMSDMetric is only implemented for input data with a dimension divisible by 3.");
        }

        float trace_a, trace_b;
        auto dim3 = static_cast<const int>(dim / 3);
        std::vector<float> buffer_a (xs, xs + dim);
        std::vector<float> buffer_b (ys, ys + dim);

        inplace_center_and_trace_atom_major(buffer_a.data(), &trace_a, 1, dim3);
        inplace_center_and_trace_atom_major(buffer_b.data(), &trace_b, 1, dim3);

        if constexpr(std::is_same<T, double>::value) {
            std::vector<float> cast (xs, xs + dim);
            return msd_atom_major(dim3, dim3, cast.data(), buffer_b.data(), trace_a, trace_b, 0, nullptr);
        } else {
            return msd_atom_major(dim3, dim3, xs, buffer_b.data(), trace_a, trace_b, 0, nullptr);
        }
    }

    template<typename T>
    static T compute(const T* xs, const T* ys, std::size_t dim) {
        return std::sqrt(compute_squared(xs, ys, dim));
    }
};

PYBIND11_MODULE(_ext, m) {
    auto rmsdModule = m.def_submodule("rmsd");
    deeptime::clustering::registerClusteringImplementation<RMSDMetric>(rmsdModule);
    rmsdModule.def("compute_metric", [](py::array_t<float> x, py::array_t<float> y) {
        return RMSDMetric::compute<float>(x.data(), y.data(), x.size());
    });
}
