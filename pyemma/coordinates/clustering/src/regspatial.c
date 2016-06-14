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

#include <clustering.h>

static PyObject *cluster(PyObject *self, PyObject *args) {
    PyObject *py_centers, *py_item, *py_res;
    PyArrayObject *np_chunk, *np_item, *np_new_center;
    Py_ssize_t N_centers, N_frames, dim, i, j, max_clusters;
    npy_intp new_dims[1];
    float *chunk;
    float **centers;
    char *metric;
    float cutoff, mindist;
    float d;
    float *buffer_a, *buffer_b;
    float (*distance)(float*, float*, size_t, float*, float*, float*);

    py_centers = NULL; py_item = NULL; py_res = NULL;
    np_chunk = NULL; np_item = NULL;
    centers = NULL; metric=""; chunk = NULL;
    buffer_a = NULL; buffer_b = NULL;

    if (!PyArg_ParseTuple(args, "O!O!fsn", &PyArray_Type, &np_chunk, &PyList_Type, &py_centers, &cutoff, &metric, &max_clusters)) goto error; /* ref:borr. */

    if(cutoff<=0.0) {
        PyErr_SetString(PyExc_ValueError, "cutoff can\'t be zero or negative.");
        goto error;
    }

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

    if(strcmp(metric,"euclidean")==0)
        distance = euclidean_distance;
    else if(strcmp(metric,"minRMSD")==0) {
        distance = minRMSD_distance;
        buffer_a = malloc(dim*sizeof(float));
        buffer_b = malloc(dim*sizeof(float));
        if(!buffer_a || !buffer_b) { PyErr_NoMemory(); goto error; }
    }
    else {
        PyErr_SetString(PyExc_ValueError, "metric must be one of \"euclidean\" or \"minRMSD\".");
        goto error;
    }

    /* import list of cluster centers */
    N_centers = PyList_Size(py_centers);
    centers = malloc(N_centers*sizeof(float*));
    if(!centers) { PyErr_NoMemory(); goto error; }

    for(i = 0; i < N_centers; ++i) {
        py_item = PyList_GetItem(py_centers,i); /* ref:borr. */
        if(!py_item) goto error;
        if(!PyArray_Check(py_item)) { PyErr_SetString(PyExc_ValueError, "Elements of centers must be numpy arrays."); goto error; }
        np_item = (PyArrayObject*)py_item;
        if(PyArray_TYPE(np_item)!=NPY_FLOAT32) { PyErr_SetString(PyExc_ValueError, "dtype of cluster center isn\'t float (32)."); goto error; };
        if(!PyArray_ISBEHAVED_RO(np_item) ) { PyErr_SetString(PyExc_ValueError, "cluster center isn\'t behaved."); goto error; };
        if(PyArray_NDIM(np_item)!=1) { PyErr_SetString(PyExc_ValueError, "Number of dimensions of cluster centers must be 1."); goto error;  };
        if(np_item->dimensions[0]!=dim) {
          PyErr_SetString(PyExc_ValueError, "Dimension of cluster centers doesn\'t match dimension of frames.");
          goto error;
        }
        centers[i] = (float*)PyArray_DATA(np_item);
    }

    /* do the clustering */
    for(i = 0; i < N_frames; ++i) {
        mindist = FLT_MAX;
        for(j = 0; j < N_centers; ++j) {
            d = distance(&chunk[i*dim], centers[j], dim, buffer_a, buffer_b, NULL);
            if(d<mindist) mindist = d;
        }
        if(mindist > cutoff) {
            if(N_centers+1>max_clusters) {
                PyErr_SetString(PyExc_RuntimeError, "Maximum number of cluster centers reached. "\
                                                    "Consider increasing max_clusters or choose "\
                                                    "a larger minimum distance, dmin.");
                goto error;
            }
            new_dims[0] = dim;
            np_new_center = (PyArrayObject*) PyArray_SimpleNew(1, new_dims, NPY_FLOAT32); /* ref:new */
            if(!np_new_center) goto error;
            memcpy(PyArray_DATA(np_new_center), &chunk[i*dim], sizeof(float)*dim);
            if(PyList_Append(py_centers, (PyObject*)np_new_center)!=0) goto error; /* ref:not stolen */
            Py_DECREF(np_new_center);
            centers = realloc(centers, (N_centers+1)*sizeof(float*));
            if(!centers) { PyErr_NoMemory(); goto error; }
            centers[N_centers] = (float*)PyArray_DATA(np_new_center);
            N_centers++;
        }
    }

    py_res = Py_BuildValue(""); /* =None */
    /* fall through */
error:
    free(centers);
    free(buffer_a);
    free(buffer_b);
    return py_res;
}

static char MOD_USAGE[] = "Chunked regular spatial clustering";

static char CLUSTER_USAGE[] = "cluster(chunk, centers, mindist, metric)\n"\
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
"max_clusters : unsigned integer\n"\
"    (input) Maximum allowed number of cluster. The function will raise\n"\
"    a `RuntimeError` when this limit is exceeded.\n"\
"\n"\
"Returns\n"\
"-------\n"\
"None. The input parameter `centers` is updated instead.\n"\
"\n"\
"Note\n"\
"----\n"\
"This function uses the minRMSD implementation of mdtraj.";


static PyMethodDef regspatialMethods[] =
{
     {"cluster", cluster, METH_VARARGS, CLUSTER_USAGE},
     {"assign",  assign,  METH_VARARGS, ASSIGN_USAGE},
     {NULL, NULL, 0, NULL}
};

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

static PyObject *
error_out(PyObject *m) {
    struct module_state *st = GETSTATE(m);
    PyErr_SetString(st->error, "something bad happened");
    return NULL;
}


#if PY_MAJOR_VERSION >= 3

static int myextension_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int myextension_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}


static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "regspatial",
        NULL,
        sizeof(struct module_state),
        regspatialMethods,
        NULL,
        myextension_traverse,
        myextension_clear,
        NULL
};

#define INITERROR return NULL

PyObject *
PyInit_regspatial(void)

#else // py2
#define INITERROR return

PyMODINIT_FUNC initregspatial(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule3("regspatial", regspatialMethods, MOD_USAGE);
#endif
    struct module_state *st = GETSTATE(module);

    if (module == NULL)
        INITERROR;

    st->error = PyErr_NewException("regspatial.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }
    // numpy support
    import_array();


#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
