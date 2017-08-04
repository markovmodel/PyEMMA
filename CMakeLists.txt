cmake_minimum_required (VERSION 2.8.12)
project (pyemma)


add_subdirectory(pybind11)


pybind11_add_module(regspace_clustering pyemma/coordinates/clustering/src/clustering_module.cpp
        )

include_directories(pyemma/coordinates/clustering/include
/home/mi/marscher/miniconda/lib/python3.6/site-packages/mdtraj/core/lib/

        )
