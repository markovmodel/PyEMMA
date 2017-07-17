cmake_minimum_required (VERSION 2.8.12)
project (pyemma)

##add_subdirectory(module/core)
#
#find_program(PYTHON "python")
#
#
#
#set(SETUP_PY "${CMAKE_CURRENT_BINARY_DIR}/../setup.py")
#set(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp")
#add_custom_command(OUTPUT ${OUTPUT}
#        COMMAND ${PYTHON} ${SETUP_PY} build
#        COMMAND ${CMAKE_COMMAND} -E touch ${OUTPUT}
#        DEPENDS ${DEPS}
#        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/../"
#        )
#
#add_custom_target(target ALL DEPENDS ${OUTPUT})
#
#install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY} develop)")

set(SOURCE_FILES pyemma/coordinates/clustering/src/regspace_module.cpp)
add_subdirectory(pybind11)


pybind11_add_module(regspace_clustering pyemma/coordinates/clustering/src/regspace_module.cpp)
include_directories(pyemma/coordinates/clustering/include
/home/mi/marscher/miniconda/lib/python3.6/site-packages/mdtraj/core/lib/

        )
