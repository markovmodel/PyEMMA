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

#include <time.h>
#include <clustering.h>

static PyObject *set_callback = NULL;

static PyObject *c_set_callback(PyObject *dummy, PyObject *args) {
    PyObject *result;
    PyObject *temp;

    result = NULL;
    temp = NULL;

    if (PyArg_ParseTuple(args, "O:set_callback", &temp)) {
        if (!PyCallable_Check(temp)) {
            PyErr_SetString(PyExc_TypeError, "parameter must be callable");
            return NULL;
        }
        /* Add a reference to new callback */
        Py_XINCREF(temp);
        /* Dispose of previous callback */
        Py_XDECREF(set_callback);
        /* Remember new callback */
        set_callback = temp;
        /* Boilerplate to return "None" */
        Py_INCREF(Py_None);
        result = Py_None;
    }
    return result;
}

static PyObject *cluster(PyObject *self, PyObject *args) {
    int debug;
    PyObject *py_centers, *py_item, *py_res;
    PyArrayObject *np_chunk, *np_item;
    Py_ssize_t N_centers, N_frames, dim;
    float *chunk;
    float **centers;
    char *metric;
    float mindist;
    float d;
    float *buffer_a, *buffer_b;
    int l;
    int *centers_counter;
    void *arr_data;
    float *new_centers;
    int i, j;
    int closest_center_index;
    npy_intp dims[2];
    float (*distance)(float*, float*, size_t, float*, float*);
    PyObject* return_new_centers;
    debug = 0;
    if(debug) printf("KMEANS: \n----------- cluster called ----------\n");
    if(debug) printf("KMEANS: declaring variables...");
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
    for(i = 0; i < N_centers; ++i) {
        l = 0;
        if(debug) printf("%d", l++); /* 0 */
        py_item = PyList_GetItem(py_centers,i); /* ref:borr. */
        if(debug) printf("%d", l++); /* 1 */
        if(!py_item) goto error;
        if(debug) printf("%d", l++); /* 2 */
        if(!PyArray_Check(py_item)) { PyErr_SetString(PyExc_ValueError, "Elements of centers must be numpy arrays."); goto error; }
        if(debug) printf("%d", l++); /* 3 */
        np_item = (PyArrayObject*)py_item;
        if(debug) printf("%d", l++); /* 4 */
        if(PyArray_TYPE(np_item)!=NPY_FLOAT32) { PyErr_SetString(PyExc_ValueError, "dtype of cluster center isn\'t float (32)."); goto error; };
        if(debug) printf("%d", l++); /* 5 */
        if(!PyArray_ISBEHAVED_RO(np_item) ) { PyErr_SetString(PyExc_ValueError, "cluster center isn\'t behaved."); goto error; };
        if(debug) printf("%d", l++); /* 6 */
        if(PyArray_NDIM(np_item)!=1) { PyErr_SetString(PyExc_ValueError, "Number of dimensions of cluster centers must be 1."); goto error;  };
        if(debug) printf("%d", l++); /* 7 */
        if(np_item->dimensions[0]!=dim) {
          PyErr_SetString(PyExc_ValueError, "Dimension of cluster centers doesn\'t match dimension of frames.");
          goto error;
        }
        if(debug) printf("%d", l++); /* 8 */
        centers[i] = (float*)PyArray_DATA(np_item);
        if(debug) printf("%d", l++); /* 9 */
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
            if(d<mindist) {
                mindist = d;
                closest_center_index = j;
            }
        }
        (*(centers_counter + closest_center_index))++;
	    for (j = 0; j < dim; j++) {
	        new_centers[closest_center_index*dim+j] += chunk[i*dim+j];
	    }
    }

    for (i = 0; i < N_centers; i++) {
        if (*(centers_counter + i) == 0) {
            for (j = 0; j < dim; j++) {
                (*(new_centers + i * dim + j)) = centers[i][j];
            }
        } else {
            for (j=0; j < dim; j++) {
                (*(new_centers + i * dim + j)) /= (*(centers_counter + i));
            }
        }
    }
    if(debug) printf("done\n");

    if(debug) printf("KMEANS: creating return_new_centers...");
    dims[0] = N_centers; dims[1] = dim;
    return_new_centers = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (return_new_centers == NULL){
        PyErr_SetString(PyExc_MemoryError, "Error occurs when creating a new PyArray");
        goto error;
    }
    arr_data = PyArray_DATA((PyArrayObject*)return_new_centers);
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

static PyObject* costFunction(PyObject *self, PyObject *args) {
    int k, i, j, r;
    float value, d;
    float *data, *centers;
    char *metric;
    PyObject *ret_cost;
    Py_ssize_t dim, n_frames;
    PyArrayObject *np_data, *np_centers;
    float (*distance)(float*, float*, size_t, float*, float*);
    float *buffer_a, *buffer_b;

    k = 0; r = 0; i = 0; j = 0; value = 0.0; d = 0.0;
    metric = NULL; np_data = NULL;
    data = NULL; ret_cost = Py_BuildValue("");
    buffer_a = NULL; buffer_b = NULL;
    /* parse python input (np_data, np_centers, metric, k) */
    if (!PyArg_ParseTuple(args, "O!O!si", &PyArray_Type, &np_data, &PyList_Type, &np_centers, &metric, &k)) {
        goto error;
    }
    n_frames = np_data->dimensions[0];
    dim = np_data->dimensions[1];
    data = PyArray_DATA(np_data);
    /* parse and initialize metric */
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

    for(r = 0; r < k; r++) {
        centers = PyArray_DATA(PyList_GetItem(np_centers,r));
        for(i = 0; i < n_frames; i++) {
            value += pow(distance(&data[i*dim], &centers[0], dim, buffer_a, buffer_b), 2);
        }
    }
    ret_cost = Py_BuildValue("f", value);
    Py_INCREF(ret_cost);
error:
    return ret_cost;
}

static PyObject* initCentersKMpp(PyObject *self, PyObject *args) {
    int k, centers_found, first_center_index, i, j, n_trials;
    int some_not_done;
    float d;
    float dist_sum;
    float sum;
    Py_ssize_t dim, n_frames;
    PyObject *ret_init_centers;
    PyArrayObject *np_data;
    PyObject *py_callback_result;
    char *metric;
    npy_intp dims[2];
    int *taken_points;
    int best_candidate = -1;
    float best_potential = FLT_MAX;
    int *next_center_candidates;
    float *next_center_candidates_rand;
    float *next_center_candidates_potential;
    float *data, *init_centers;
    float *buffer_a, *buffer_b;
    void *arr_data;
    float *squared_distances;
    float (*distance)(float*, float*, size_t, float*, float*);

    ret_init_centers = Py_BuildValue("");
    py_callback_result = NULL;
    np_data = NULL; metric = NULL; data = NULL;
    init_centers = NULL; taken_points = NULL;
    centers_found = 0; squared_distances = NULL;
    buffer_a = NULL; buffer_b = NULL;
    next_center_candidates = NULL;
    next_center_candidates_rand = NULL;
    next_center_candidates_potential = NULL;
    dist_sum = 0.0;


#ifndef _KMEANS_INIT_RANDOM_SEED
#define _KMEANS_INIT_RANDOM_SEED
    /* set random seed */
    srand(time(NULL));
#endif

    /* parse python input (np_data, metric, k) */
    if (!PyArg_ParseTuple(args, "O!si", &PyArray_Type, &np_data, &metric, &k)) {
        goto error;
    }
    n_frames = np_data->dimensions[0];
    dim = np_data->dimensions[1];
    data = PyArray_DATA(np_data);
    /* number of trials before choosing the data point with the best potential */
    n_trials = 2 + (int) log(k);

    /* allocate space for the index giving away which point has already been used as a cluster center */
    if(!(taken_points = (int*) calloc(n_frames, sizeof(int)))) { PyErr_NoMemory(); goto error; }
    /* allocate space for the array holding the cluster centers to be returned */
    if(!(init_centers = (float*) calloc(k * dim, sizeof(float)))) { PyErr_NoMemory(); goto error; }
    /* allocate space for the array holding the squared distances to the assigned cluster centers */
    if(!(squared_distances = (float*) calloc(n_frames, sizeof(float)))) { PyErr_NoMemory(); goto error; }

    /* candidates allocations */
    if(!(next_center_candidates = (int*) malloc(n_trials * sizeof(int)))) { PyErr_NoMemory(); goto error; }
    if(!(next_center_candidates_rand = (float*) malloc(n_trials * sizeof(float)))) { PyErr_NoMemory(); goto error; }
    if(!(next_center_candidates_potential = (float*) malloc(n_trials * sizeof(float)))) { PyErr_NoMemory(); goto error; }

    /* parse and initialize metric */
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

    /* pick first center randomly */
    first_center_index = rand() % n_frames;
    /* and mark it as assigned */
    taken_points[first_center_index] = 1;
    /* write its coordinates into the init_centers array */
    for(j = 0; j < dim; j++) {
        (*(init_centers + centers_found*dim + j)) = data[first_center_index*dim + j];
    }
    /* increase number of found centers */
    centers_found++;
    /* perform callback */
    if(set_callback) {
        py_callback_result = PyObject_CallObject(set_callback, NULL);
        if(py_callback_result) Py_DECREF(py_callback_result);
    }

    /* iterate over all data points j, measuring the squared distance between j and the initial center i: */
    /* squared_distances[i] = distance(x_j, x_i)*distance(x_j, x_i) */
    for(i = 0; i < n_frames; i++) {
        if(i != first_center_index) {
            d = pow(distance(&data[i*dim], &data[first_center_index*dim], dim, buffer_a, buffer_b), 2);
            squared_distances[i] = d;
            /* build up dist_sum which keeps the sum of all squared distances */
            dist_sum += d;
        }
    }

    /* keep picking centers while we do not have enough of them... */
    while(centers_found < k) {

        /* initialize the trials random values by the D^2-weighted distribution */
        for(j = 0; j < n_trials; j++) {
            next_center_candidates[j] = -1;
            next_center_candidates_rand[j] = dist_sum * ((float)rand()/(float)RAND_MAX);
            next_center_candidates_potential[j] = 0.0;
        }

        /* pick candidate data points corresponding to their random value */
        sum = 0.0;
        for(i = 0; i < n_frames; i++) {
            if (!taken_points[i]) {
                sum += squared_distances[i];
                some_not_done = 0;
                for(j = 0; j < n_trials; j++) {
                    if(next_center_candidates[j] == -1) {
                        if (sum >= next_center_candidates_rand[j]) {
                            next_center_candidates[j] = i;
                        } else {
                            some_not_done = 1;
                        }
                    }
                }
                if(!some_not_done) break;
            }
        }

        /* now find the maximum squared distance for each trial... */
        for(i = 0; i < n_frames; i++) {
            if (!taken_points[i]) {
                for(j = 0; j < n_trials; j++) {
                    if(next_center_candidates[j] == -1) break;
                    if(next_center_candidates[j] != i) {
                        d = pow(distance(&data[i*dim], &data[next_center_candidates[j]*dim], dim, buffer_a, buffer_b), 2);
                        if(d < squared_distances[i]) {
                            next_center_candidates_potential[j] += d;
                        } else {
                            next_center_candidates_potential[j] += squared_distances[i];
                        }
                    }
                }
            }
        }

        /* ... and select the best candidate by the minimum value of the maximum squared distances */
        best_candidate = -1;
        best_potential = FLT_MAX;
        for(j = 0; j < n_trials; j++) {
            if(next_center_candidates[j] != -1 && next_center_candidates_potential[j] < best_potential) {
                 best_potential = next_center_candidates_potential[j];
                 best_candidate = next_center_candidates[j];
            }
        }

        /* if for some reason we did not find a best candidate, just take the next available point */
        if(best_candidate == -1) {
            for(i = 0; i < n_frames; i++) {
                if(!taken_points[i]) {
                    best_candidate = i;
                    break;
                }
            }
        }

        /* check if best_candidate was set, otherwise break to avoid an infinite loop should things go wrong */
        if(best_candidate >= 0) {
            /* write the best_candidate's components into the init_centers array */
            for(j = 0; j < dim; j++) {
                (*(init_centers + centers_found*dim + j)) = (*(data + best_candidate*dim + j));
            }
            /* increase centers_found */
            centers_found++;
            /* perform the callback */
            if(set_callback) {
                py_callback_result = PyObject_CallObject(set_callback, NULL);
                if(py_callback_result) Py_DECREF(py_callback_result);
            }
            /* mark the data point as assigned center */
            taken_points[best_candidate] = 1;
            /* update the sum of squared distances by removing the assigned center */
            dist_sum -= squared_distances[best_candidate];

            /* if we still have centers to assign, the squared distances array has to be updated */
            if(centers_found < k) {
                /* Check for each data point if its squared distance to the freshly added center is smaller than */
                /* the squared distance to the previously picked centers. If so, update the squared_distances */
                /* array by the new value and also update the dist_sum value by removing the old value and adding */
                /* the new one. */
                for(i = 0; i < n_frames; i++) {
                    if(!taken_points[i]) {
                        d = pow(distance(&data[i*dim], &data[best_candidate*dim], dim, buffer_a, buffer_b), 2);
                        if(d < squared_distances[i]) {
                            dist_sum += d - squared_distances[i];
                            squared_distances[i] = d;
                        }
                    }
                }
            }
        } else {
            break;
        }
    }

    /* create the output objects */
    dims[0] = k;
    dims[1] = dim;
    ret_init_centers = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (ret_init_centers == NULL){
        PyErr_SetString(PyExc_MemoryError, "Error occurs when creating a new PyArray");
        goto error;
    }
    arr_data = PyArray_DATA((PyArrayObject*)ret_init_centers);
    /* Need to copy the data of the malloced buffer to the PyObject
       since the malloced buffer will disappear after the C extension is called. */
    memcpy(arr_data, init_centers, PyArray_ITEMSIZE((PyArrayObject*) ret_init_centers) * k * dim);
    Py_INCREF(ret_init_centers);  /* The returned list should still exist after calling the C extension */
error:
    free(buffer_a);
    free(buffer_b);
    free(taken_points);
    free(init_centers);
    free(squared_distances);
    free(next_center_candidates);
    free(next_center_candidates_rand);
    free(next_center_candidates_potential);
    return ret_init_centers;
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

#define INIT_CENTERS_USAGE "init_centers(data, metric, k)\n"\
"Given the data, choose \"k\" cluster centers according to the kmeans++ initialization."\
"\n"\
"Parameters\n"\
"----------\n"\
"data : (N,M) C-style contiguous and behaved ndarray of np.float32\n"\
"    (input) array of data points, each having dimension M.\n"\
"metric : string\n"\
"    (input) One of \"euclidean\" or \"minRMSD\" (case sensitive).\n"\
"k : int\n"\
"    (input) the number of cluster centers to be assigned for initialization."\
"\n"\
"Returns\n"\
"-------\n"\
"A numpy ndarray of cluster centers assigned to the provided data set.\n"\
"\n"\
"Note\n"\
"----\n"\
"This function uses the minRMSD implementation of mdtraj."


static PyMethodDef kmeansMethods[] =
{
     {"cluster", cluster, METH_VARARGS, CLUSTER_USAGE},
     {"assign",  assign,  METH_VARARGS, ASSIGN_USAGE},
     {"init_centers", initCentersKMpp, METH_VARARGS, INIT_CENTERS_USAGE},
     {"cost_function", costFunction, METH_VARARGS, "Evaluates the cost function for the k-means clustering algorithm."},
     {"set_callback", c_set_callback, METH_VARARGS, "For setting a callback."},
     {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initkmeans_clustering(void)
{
  (void)Py_InitModule3("kmeans_clustering", kmeansMethods, MOD_USAGE);
  import_array();
}