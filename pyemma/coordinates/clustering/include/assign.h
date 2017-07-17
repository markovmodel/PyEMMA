//
// Created by marscher on 7/17/17.
//

#ifndef PYEMMA_ASSIGN_H
#define PYEMMA_ASSIGN_H

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "metric_base.h"

namespace py = pybind11;

template <typename dtype>
int c_assign(py::array_t<dtype> chunk, py::list centers, py::array_t<int> dtraj, metric::metric_base* metric_,
             int dim, int n_threads) {
    int ret;
    int debug;
    Py_ssize_t i, j;
    float d, mindist, trace_centers;
    size_t argmin;
    float *buffer_a, *buffer_b;
    float *centers_precentered;
    float *trace_centers_p;
    float *dists;
    // distance function pointer:
    float (*distance)(float*, float*, size_t, float*, float*, float*);
    float *SKP_restrict chunk_p;

#ifdef USE_OPENMP
    /* Create a parallel thread block. */
    omp_set_num_threads(n_threads);
    if(debug) printf("using openmp; n_threads=%i\n", n_threads);
    assert(omp_get_num_threads() == n_threads);
#endif

    /* Initialize variables */
    buffer_a = NULL; buffer_b = NULL; trace_centers_p = NULL; centers_precentered = NULL;
    chunk_p = NULL; dists = NULL;
    ret = ASSIGN_SUCCESS;
    debug=0;

    /* init metric */
    if(strcmp(metric, "euclidean")==0) {
        distance = euclidean_distance;
        dists = malloc(N_centers*sizeof(float));
        if(!dists) {
            ret = ASSIGN_ERR_NO_MEMORY;
        }
    } else if(strcmp(metric, "minRMSD")==0) {
        distance = minRMSD_distance;
        centers_precentered = malloc(N_centers*dim*sizeof(float));
        trace_centers_p = malloc(N_centers*sizeof(float));
        dists = malloc(N_centers*sizeof(float));
        if(!centers_precentered || !dists || !trace_centers_p) {
            ret = ASSIGN_ERR_NO_MEMORY;
        }

        if (ret == ASSIGN_SUCCESS) {
            memcpy(centers_precentered, centers, N_centers*dim*sizeof(float));

            /* Parallelize centering of cluster generators */
            /* Note that this is already OpenMP-enabled */
            for (j = 0; j < N_centers; ++j) {
                inplace_center_and_trace_atom_major(&centers_precentered[j*dim],
                                                    &trace_centers_p[j], 1, dim/3);
            }
            centers = centers_precentered;
        }
    } else {
        ret = ASSIGN_ERR_INVALID_METRIC;
    }

#pragma omp parallel private(buffer_a, buffer_b, i, j, chunk_p, mindist, argmin)
    {
        /* Allocate thread storage */
        buffer_a = malloc(dim*sizeof(float));
        buffer_b = malloc(dim*sizeof(float));
#pragma omp critical
        if(!buffer_a || !buffer_b) {
            ret = ASSIGN_ERR_NO_MEMORY;
        }
#pragma omp barrier
#pragma omp flush(ret)

        /* Only proceed if no error occurred. */
        if (ret == ASSIGN_SUCCESS) {

            /* Assign each frame */
            for(i = 0; i < N_frames; ++i) {
                chunk_p = &chunk[i*dim];

                /* Parallelize distance calculations to cluster centers to avoid cache misses */
#pragma omp for
                for(j = 0; j < N_centers; ++j) {
                    dists[j] = distance(&centers[j*dim], chunk_p, dim, buffer_a, buffer_b, &trace_centers_p[j]);
                }
#pragma omp flush(dists)

                /* Only one thread can make actual assignment */
#pragma omp single
                {
                    mindist = FLT_MAX; argmin = -1;
                    for (j=0; j < N_centers; ++j) {
                        if (dists[j] < mindist) { mindist = dists[j]; argmin = j; }
                    }
                    dtraj[i] = argmin;
                }

                /* Have all threads synchronize in progress through cluster assignments */
#pragma omp barrier
            }

            /* Clean up thread storage*/
            free(buffer_a);
            free(buffer_b);
        }
    }

    /* Clean up global storage */
    if (dists) free(dists);
    if (centers_precentered) free(centers_precentered);
    if (trace_centers_p) free(trace_centers_p);
    return ret;
}
#endif //PYEMMA_ASSIGN_H
