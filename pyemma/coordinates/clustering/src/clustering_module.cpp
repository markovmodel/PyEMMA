//
// Created by marscher on 7/17/17.
//

#include <center.h>
#include <theobald_rmsd.h>
#include <deeptime/clustering/register_clustering.h>

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
            return msd_atom_major(dim3, dim3, buffer_a.data(), buffer_b.data(), trace_a, trace_b, 0, nullptr);
        }
    }

    template<typename T>
    static T compute(const T* xs, const T* ys, std::size_t dim) {
        return std::sqrt(compute_squared(xs, ys, dim));
    }

    /*template<typename T>
    static void precenter_centers(np_array_nfc<T> centers, std::size_t N_centers) {
        trace_centers.resize(N_centers);
        float *trace_centers_p = trace_centers.data();

        // This is already parallelized
        for (std::size_t j = 0; j < N_centers; ++j) {
            inplace_center_and_trace_atom_major(&centers[j * parent_t::dim],
                                                &trace_centers_p[j], 1, parent_t::dim / 3);
        }
    }*/
};

PYBIND11_MODULE(_ext, m) {
    auto rmsdModule = m.def_submodule("rmsd");
    deeptime::clustering::registerClusteringImplementation<RMSDMetric>(rmsdModule);
}
