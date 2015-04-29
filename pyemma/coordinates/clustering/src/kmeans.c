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

#include <clustering.h>

static PyObject *cluster(PyObject *self, PyObject *args) {
    int debug;
    debug = 0;
    if(debug) printf("KMEANS: \n----------- cluster called ----------\n");
    if(debug) printf("KMEANS: declaring variables...");
    PyObject *py_centers, *py_item, *py_res;
    PyArrayObject *np_chunk, *np_item;
    Py_ssize_t N_centers, N_frames, dim;
    float *chunk;
    float **centers;
    char *metric;
    float mindist;
    float d;
    float *buffer_a, *buffer_b;
    int *centers_counter;
    float *new_centers;
    int i, j;
    int closest_center_index;
    float (*distance)(float*, float*, size_t, float*, float*);
    PyObject* return_new_centers;
    if(debug) printf("done.\n");

    if(debug) printf("KMEANS: initializing some of them...");
    py_centers = NULL; py_item = NULL; py_res = NULL;
    np_chunk = NULL; np_item = NULL;
    centers = NULL; metric=""; chunk = NULL;
    centers_counter = NULL; new_centers = NULL;
    buffer_a = NULL; buffer_b = NULL;
    return_new_centers = Py_BuildValue("");
    if(debug) printf("done\n");

    if(debug) printf("KMEANS: attempting to parse args...");
    if (!PyArg_ParseTuple(args, "O!O!s", &PyArray_Type, &np_chunk, &PyList_Type, &py_centers, &metric)) {
        goto error;
    }
    if(debug) printf("done\n");

    /* import chunk */
    if(debug) printf("KMEANS: importing chunk...");
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
    if(debug) printf("done with N_frames=%zd, dim=%zd\n", N_frames, dim);

    if(debug) printf("KMEANS: creating metric function...");
    if(strcmp(metric,"euclidean")==0) {
        distance = euclidean_distance;
    } else if(strcmp(metric,"minRMSD")==0) {
        distance = minRMSD_distance;
        buffer_a = malloc(dim*sizeof(float));
        buffer_b = malloc(dim*sizeof(float));
        if(!buffer_a || !buffer_b) { PyErr_NoMemory(); goto error; }
    } else {
        PyErr_SetString(PyExc_ValueError, "metric must be one of \"euclidean\" or \"minRMSD\".");
        goto error;
    }
    if(debug) printf("done\n");

    /* import list of cluster centers */
    if(debug) printf("KMEANS: importing list of cluster centers...");
    N_centers = PyList_Size(py_centers);
    if(!(centers = malloc(N_centers*sizeof(float*)))) {
        PyErr_NoMemory(); goto error;
    }
    int l;
    for(i = 0; i < N_centers; ++i) {
        l = 0;
        if(debug) printf("%d", l++); // 0
        py_item = PyList_GetItem(py_centers,i); /* ref:borr. */
        if(debug) printf("%d", l++); // 1
        if(!py_item) goto error;
        if(debug) printf("%d", l++); // 2
        if(!PyArray_Check(py_item)) { PyErr_SetString(PyExc_ValueError, "Elements of centers must be numpy arrays."); goto error; }
        if(debug) printf("%d", l++); // 3
        np_item = (PyArrayObject*)py_item;
        if(debug) printf("%d", l++); // 4
        if(PyArray_TYPE(np_item)!=NPY_FLOAT32) { PyErr_SetString(PyExc_ValueError, "dtype of cluster center isn\'t float (32)."); goto error; };
        if(debug) printf("%d", l++); // 5
        if(!PyArray_ISBEHAVED_RO(np_item) ) { PyErr_SetString(PyExc_ValueError, "cluster center isn\'t behaved."); goto error; };
        if(debug) printf("%d", l++); // 6
        if(PyArray_NDIM(np_item)!=1) { PyErr_SetString(PyExc_ValueError, "Number of dimensions of cluster centers must be 1."); goto error;  };
        if(debug) printf("%d", l++); // 7
        if(np_item->dimensions[0]!=dim) {
          PyErr_SetString(PyExc_ValueError, "Dimension of cluster centers doesn\'t match dimension of frames.");
          goto error;
        }
        if(debug) printf("%d", l++); // 8
        centers[i] = (float*)PyArray_DATA(np_item);
        if(debug) printf("%d", l++); // 9
    }
    if(debug) printf("done, k=%zd\n", N_centers);

    /* initialize centers_counter and new_centers with zeros */
    if(debug) printf("KMEANS: initializing calloc centers counter and new_centers stuff...");
    centers_counter = (int*) calloc(N_centers, sizeof(int));
    new_centers = (float*) calloc(N_centers * dim, sizeof(float));
    if(!centers_counter || !new_centers) { PyErr_NoMemory(); goto error; }
    if(debug) printf("done\n");

    /* do the clustering */
    if(debug) printf("KMEANS: performing the clustering...");
    for (i = 0; i < N_frames; i++) {
        mindist = FLT_MAX;
        for(j = 0; j < N_centers; ++j) {
            d = distance(&chunk[i*dim], centers[j], dim, buffer_a, buffer_b);
            //printf("\t\t\td=%.6f, &chunk[i*dim]=%.6f\n", d, chunk[i*dim]);
            if(d<mindist) {
                mindist = d;
                closest_center_index = j;
            }
        }
        (*(centers_counter + closest_center_index))++;
	    for (j = 0; j < dim; j++) {
	        //printf("\t\ttest=%.6f,\t idx=%d\n", chunk[i*dim+j], closest_center_index*dim+j);
	        new_centers[closest_center_index*dim+j] += chunk[i*dim+j];
	        //(*(new_centers + closest_center_index * dim + j)) += chunk[i*dim+j];
	    }
    }

    for (i = 0; i < N_centers; i++) {
        if (*(centers_counter + i) == 0) {
            for (j = 0; j < dim; j++) {
                (*(new_centers + i * dim + j)) = centers[i][j];//(*(centers + i*dim + j));
            }
        } else {
            for (j=0; j < dim; j++) {
                (*(new_centers + i * dim + j)) /= (*(centers_counter + i));
            }
        }
    }
    if(debug) printf("done\n");

    if(debug) printf("KMEANS: creating return_new_centers...");
    npy_intp dims[2] = {N_centers, dim};
    return_new_centers = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (return_new_centers == NULL){
        PyErr_SetString(PyExc_MemoryError, "Error occurs when creating a new PyArray");
        goto error;
    }
    void *arr_data = PyArray_DATA((PyArrayObject*)return_new_centers);
    if(debug) printf("done\n");
    /* Need to copy the data of the malloced buffer to the PyObject
       since the malloced buffer will disappear after the C extension is called. */
    if(debug) printf("KMEANS: attempting memcopy...");
    memcpy(arr_data, new_centers, PyArray_ITEMSIZE((PyArrayObject*) return_new_centers) * N_centers * dim);
    if(debug) printf("done\n");
    if(debug) printf("KMEANS: increasting ref to return_new_centers thingy...");
    Py_INCREF(return_new_centers);  /* The returned list should still exist after calling the C extension */
    if(debug) printf("done\n");
    /* fall through */
error:
    free(centers_counter);
    free(new_centers);
    free(centers);
    free(buffer_a);
    free(buffer_b);
    return return_new_centers;
}

#define MOD_USAGE "Chunked regular spatial clustering"

#define CLUSTER_USAGE "cluster(chunk, centers, mindist, metric)\n"\
"Given a chunk of data and a list of cluster centers, update the list of cluster centers with the newly found centers.\n"\
"\n"\
"Parameters\n"\
"----------\n"\
"chunk : (N,M) C-style contiguous and behaved ndarray of np.float32\n"\
"    (input) array of N frames, each frame having dimension M\n"\
"centers : list of (M) behaved ndarrays of np.float32\n"\
"    (input/output) Possibly empty list of previously found cluster\n"\
"    centers. New centers are appended to this list.\n"\
"dmin : float\n"\
"    (input) Distance parameter for regular spatial clustering. Whenever\n"\
"    a frame is at least `dmin` away form all cluster centers it is added\n"\
"    as a new cluster center.\n"\
"metric : string\n"\
"    (input) One of \"euclidean\" or \"minRMSD\" (case sensitive).\n"\
"\n"\
"Returns\n"\
"-------\n"\
"None. The input parameter `centers` is updated instead.\n"\
"\n"\
"Note\n"\
"----\n"\
"This function uses the minRMSD implementation of mdtraj."


static PyMethodDef kmeansMethods[] =
{
     {"cluster", cluster, METH_VARARGS, CLUSTER_USAGE},
     {"assign",  assign,  METH_VARARGS, ASSIGN_USAGE},
     {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initkmeans_clustering(void)
{
  (void)Py_InitModule3("kmeans_clustering", kmeansMethods, MOD_USAGE);
  import_array();
}