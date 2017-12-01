cmake_minimum_required (VERSION 2.8.12)
project (pyemma)


add_subdirectory(pybind11)


pybind11_add_module(regspace_clustering pyemma/_ext/variational/estimators/covar_c/covartools.cpp
        )

include_directories(pyemma/_ext/variational/covar_c/

        )
