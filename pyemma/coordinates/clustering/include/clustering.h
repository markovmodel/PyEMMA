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
float euclidean_distance(float *a, float *b, size_t n, float *buffer_a, float *buffer_b);
// minRMSD metric
float minRMSD_distance(float *a, float *b, size_t n, float *buffer_a, float *buffer_b);

// assignment to cluster centers
PyObject *assign(PyObject *self, PyObject *args);
//void c_assign(float *chunk, float *centers, char *metric, )

#ifdef __cplusplus
}
#endif
#endif