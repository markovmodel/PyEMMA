/* * Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
 * Berlin, 14195 Berlin, Germany.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation and/or
 * other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 * ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _CLUSTERING_H_
#define _CLUSTERING_H_
#ifdef __cplusplus
extern "C" {
#endif

#include <Python.h>
#define PY_ARRAY_UNIQUE_SYMBOL pyemma_clustering_ARRAY_API
#include <numpy/arrayobject.h>

#ifdef _OPENMP
#include <omp.h>
#endif /* _OPENMP */

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <theobald_rmsd.h>
#include <center.h>
#include <stdio.h>
#include <float.h>

#if defined(__GNUC__) && ((__GNUC__ > 3) || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#   define SKP_restrict __restrict
#elif defined(_MSC_VER) && _MSC_VER >= 1400
#   define SKP_restrict __restrict
#else
#   define SKP_restrict
#endif

#define ASSIGN_SUCCESS 0
#define ASSIGN_ERR_NO_MEMORY 1
#define ASSIGN_ERR_INVALID_METRIC 2

#define ASSIGN_USAGE "assign(chunk, centers, dtraj, metric)\n"\
"Assigns frames in `chunk` to the closest cluster centers.\n"\
"\n"\
"Parameters\n"\
"----------\n"\
"chunk : (N,M) C-style contiguous and behaved ndarray of np.float32\n"\
"    (input) array of N frames, each frame having dimension M\n"\
"centers : (M,K) ndarray-like of np.float32\n"\
"    (input) Non-empty array-like of cluster centers.\n"\
"dtraj : (N) ndarray of np.int64\n"\
"    (output) discretized trajectory\n"\
"    dtraj[i]=argmin{ d(chunk[i,:],centers[j,:]) | j in 0...(K-1) }\n"\
"    where d is the metric that is specified with the argument `metric`.\n"\
"metric : string\n"\
"    (input) One of \"euclidean\" or \"minRMSD\" (case sensitive).\n"\
"\n"\
"Returns\n"\
"-------\n"\
"None\n"\
"\n"\
"Note\n"\
"----\n"\
"This function uses the minRMSD implementation of mdtraj."

// euclidean metric
float euclidean_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n, float *buffer_a, float *buffer_b);
// minRMSD metric
float minRMSD_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n, float *SKP_restrict buffer_a, float *SKP_restrict buffer_b);

// assignment to cluster centers from python
PyObject *assign(PyObject *self, PyObject *args);
// assignment to cluster centers from c
int c_assign(float *chunk, float *centers, npy_int64 *dtraj, char* metric, Py_ssize_t N_frames, Py_ssize_t N_centers, Py_ssize_t dim);

#ifdef __cplusplus
}
#endif
#endif