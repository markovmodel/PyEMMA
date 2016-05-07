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

#define NO_IMPORT_ARRAY
#include <clustering.h>
#include <assert.h>

#ifdef USE_OPENMP
    #include <omp.h>
#endif

float euclidean_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n, float *buffer_a, float *buffer_b,
float*dummy)
{
    double sum;
    size_t i;

    sum = 0.0;
    for(i=0; i<n; ++i) {
        sum += (a[i]-b[i])*(a[i]-b[i]);
    }
    return sqrt(sum);
}

void in_place_center(float* a, size_t n) {
    float trace;
    inplace_center_and_trace_atom_major(a, &trace, 1, n/3);
}


float minRMSD_distance(float *SKP_restrict a, float *SKP_restrict b, size_t n,
                       float *SKP_restrict buffer_a, float *SKP_restrict buffer_b,
                       float* trace_a_precalc)
{
    float msd;
    float trace_a, trace_b;

    if (! trace_a_precalc) {
    	memcpy(buffer_a, a, n*sizeof(float));
    	memcpy(buffer_b, b, n*sizeof(float));

    	inplace_center_and_trace_atom_major(buffer_a, &trace_a, 1, n/3);
    	inplace_center_and_trace_atom_major(buffer_b, &trace_b, 1, n/3);

    	msd = msd_atom_major(n/3, n/3, buffer_a, buffer_b, trace_a, trace_b, 0, NULL);
    } else {
    	// only copy b, since a has been pre-centered,
        memcpy(buffer_b, b, n*sizeof(float));
        inplace_center_and_trace_atom_major(buffer_b, &trace_b, 1, n/3);

        msd = msd_atom_major(n/3, n/3, a, buffer_b, *trace_a_precalc, trace_b, 0, NULL);
    }

    return sqrt(msd);
}

int c_assign(float *chunk, float *centers, npy_int32 *dtraj, char* metric,
             Py_ssize_t N_frames, Py_ssize_t N_centers, Py_ssize_t dim, int n_threads) {
    int ret;
    int debug;
    Py_ssize_t i, j;
    float d, mindist, trace_centers;
    size_t argmin;
    float *buffer_a, *buffer_b;
    float *centers_precentered;
    float *trace_centers_p;
    float* dists;
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
        dists = malloc(N_centers*sizeof(float));
        if(!centers_precentered || !dists) {
            ret = ASSIGN_ERR_NO_MEMORY;
        }

        if (ret == ASSIGN_SUCCESS) {
            memcpy(centers_precentered, centers, N_centers*dim*sizeof(float));

	        /* Parallelize centering of cluster generators */
	        /* Note that this is already OpenMP-enabled */
	        inplace_center_and_trace_atom_major(centers_precentered, &trace_centers, 1, dim/3);
                trace_centers_p = &trace_centers;
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
                    dists[j] = distance(&centers[j*dim], chunk_p, dim, buffer_a, buffer_b, trace_centers_p);
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
    return ret;
}

PyObject *assign(PyObject *self, PyObject *args) {

    PyObject *py_centers, *py_res;
    PyArrayObject *np_chunk, *np_centers, *np_dtraj;
    Py_ssize_t N_centers, N_frames, dim;
    float *chunk;
    float *centers;
    npy_int32 *dtraj;
    char *metric;
    int n_threads;

    py_centers = NULL; py_res = NULL;
    np_chunk = NULL; np_dtraj = NULL;
    centers = NULL; metric=""; chunk = NULL; dtraj = NULL; n_threads = -1;

    if (!PyArg_ParseTuple(args, "O!OO!si", &PyArray_Type, &np_chunk, &py_centers, &PyArray_Type, &np_dtraj, &metric, &n_threads)) goto error; /* ref:borr. */

    /* import chunk */
    if(PyArray_TYPE(np_chunk)!=NPY_FLOAT32) { PyErr_SetString(PyExc_ValueError, "dtype of \"chunk\" isn\'t float (32)."); goto error; };
    if(!PyArray_ISCARRAY_RO(np_chunk) ) { PyErr_SetString(PyExc_ValueError, "\"chunk\" isn\'t C-style contiguous or isn\'t behaved."); goto error; };
    if(PyArray_NDIM(np_chunk)!=2) { PyErr_SetString(PyExc_ValueError, "Number of dimensions of \"chunk\" isn\'t 2."); goto error;  };
    N_frames = np_chunk->dimensions[0];
    dim = np_chunk->dimensions[1];
    if(dim==0) {
        PyErr_SetString(PyExc_ValueError, "chunk dimension must be larger than zero.");
        goto error;
    }
    chunk = PyArray_DATA(np_chunk);

    /* import dtraj */
    if(PyArray_TYPE(np_dtraj)!=NPY_INT32) { PyErr_SetString(PyExc_ValueError, "dtype of \"dtraj\" isn\'t int (32)."); goto error; };
    if(!PyArray_ISBEHAVED_RO(np_dtraj) ) { PyErr_SetString(PyExc_ValueError, "\"dtraj\" isn\'t behaved."); goto error; };
    if(PyArray_NDIM(np_dtraj)!=1) { PyErr_SetString(PyExc_ValueError, "Number of dimensions of \"dtraj\" isn\'t 1."); goto error; };
    if(np_chunk->dimensions[0]!=N_frames) {
        PyErr_SetString(PyExc_ValueError, "Size of \"dtraj\" differs from number of frames in \"chunk\".");
        goto error;
    }
    dtraj = (npy_int32*)PyArray_DATA(np_dtraj);

    /* import list of cluster centers */
    np_centers = (PyArrayObject*)PyArray_ContiguousFromAny(py_centers, NPY_FLOAT32, 2, 2);
    if(!np_centers) {
        PyErr_SetString(PyExc_ValueError, "Could not convert \"centers\" to two-dimensional C-contiguous behaved ndarray of float (32).");
        goto error;
    }
    N_centers = np_centers->dimensions[0];
    if(N_centers==0) {
        PyErr_SetString(PyExc_ValueError, "centers must contain at least one element.");
        goto error;
    }
    if(np_centers->dimensions[1]!=dim) {
        PyErr_SetString(PyExc_ValueError, "Dimension of cluster centers doesn\'t match dimension of frames.");
        goto error;
    }
    centers = (float*)PyArray_DATA(np_centers);

    /* do the assignment */
    switch(c_assign(chunk, centers, dtraj, metric, N_frames, N_centers, dim, n_threads)) {
        case ASSIGN_ERR_INVALID_METRIC:
            PyErr_SetString(PyExc_ValueError, "metric must be one of \"euclidean\" or \"minRMSD\".");
            goto error;
        case ASSIGN_ERR_NO_MEMORY:
            PyErr_NoMemory();
            goto error;
    }

    py_res = Py_BuildValue(""); /* =None */
    /* fall through */
error:
    return py_res;
}
