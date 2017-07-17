//
// Created by marscher on 7/17/17.
//

#include "regspace.h"


PYBIND11_PLUGIN(regspace_clustering) {
    py::module m("regspace_clustering");

    typedef RegularSpaceClustering<double> regspace_d;
    py::class_<regspace_d>(m, "Regspace_d_euclid")
            .def(py::init<double, std::size_t,
                    const std::string &, size_t>())
            .def("cluster", &regspace_d::cluster);

    return m.ptr();
}