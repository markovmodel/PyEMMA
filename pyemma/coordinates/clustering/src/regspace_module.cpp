//
// Created by marscher on 7/17/17.
//

#include "regspace.h"


PYBIND11_PLUGIN(regspace_clustering) {
    py::module m("regspace_clustering");

    typedef ClusteringBase<float> cbase_f;
    typedef RegularSpaceClustering<float> regspace_f;

    // we need to pass RegspaceClusterings base to pybind.
    py::class_<cbase_f>(m, "")
            .def(py::init<const std::string&, std::size_t>());

    py::class_<regspace_f, cbase_f>(m, "Regspace_f")
            .def(py::init<double, std::size_t,
                    const std::string &, size_t>())
            .def("cluster", &regspace_f::cluster)
            // todo: this should be member of the base class and callable from python as well..
            .def("assign", &regspace_f::assign_chunk_to_centers);//, "", py::arg("i") = 1);

    return m.ptr();
}