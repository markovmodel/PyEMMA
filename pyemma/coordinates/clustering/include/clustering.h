/* * This file is part of PyEMMA.
 *
 * Copyright (c) 2015, 2014 Computational Molecular Biology Group
 *
 * PyEMMA is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
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

static char ASSIGN_USAGE[] = "assign(chunk, centers, dtraj, metric)\n"\
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
"Returns \n"\
"-------\n"\
"None\n"\
"\n"\
"Note\n"\
"----\n"\
"This function uses the minRMSD implementation of mdtraj.";

// euclidean metric
float euclidean_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n, float *buffer_a, float *buffer_b, float*dummy);
// minRMSD metric
float minRMSD_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n, float *SKP_restrict buffer_a, float *SKP_restrict buffer_b,
float* pre_calc_trace_a);

// assignment to cluster centers from python
PyObject *assign(PyObject *self, PyObject *args);
// assignment to cluster centers from c
int c_assign(float *chunk, float *centers, npy_int32 *dtraj, char* metric,
             Py_ssize_t N_frames, Py_ssize_t N_centers, Py_ssize_t dim, int n_threads);

#ifdef __cplusplus
}
#endif
#endif
