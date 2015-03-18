#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <numpy/arrayobject.h>
#include <theobald_rmsd.h>
#include <center.h>
#include <stdio.h>
#include <float.h>

static float euclidean_distance(float *a, float *b, size_t n, float *buffer_a, float *buffer_b)
{
    double sum;
    size_t i;
    
    sum = 0.0;
    for(i=0; i<n; ++i) {
        sum += (a[i]-b[i])*(a[i]-b[i]);
    }
    return sqrt(sum);
}

static float minRMSD_distance(float *a, float *b, size_t n, float *buffer_a, float *buffer_b)
{
    float msd;
    float trace_a, trace_b;
    size_t i;
    
    for(i=0; i<n; ++i) buffer_a[i] = (float)a[i];
    for(i=0; i<n; ++i) buffer_b[i] = (float)b[i];
    
    inplace_center_and_trace_atom_major(buffer_a, &trace_a, 1, n/3);
    inplace_center_and_trace_atom_major(buffer_b, &trace_b, 1, n/3);
    msd = msd_atom_major(n/3, n/3, buffer_a, buffer_b, trace_a, trace_b, 0, NULL); 
    return sqrt(msd);
}

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
    float (*distance)(float*, float*, size_t, float*, float*);

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
            d = distance(&chunk[i*dim], centers[j], dim, buffer_a, buffer_b);
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

/* TODO: assign */ 

#define MOD_USAGE "Chunked regular spatial clustering"

#define CLUSTER_USAGE "cluster(chunk, centers, mindist, metric)\n"\
"Given a chunk of data and a list of cluster centers, update the list of cluster centers with the newly found centers.\n"\
"\n"\
"Parameters\n"\
"----------\n"\
"chunk : (N,M) C-style contiguous and bahaved ndarray of np.float32\n"\
"    array of N frames, each frame having dimension M\n"\
"centers : list of (M) behaved ndarrays of np.float32\n"\
"    Possibly empty list of previously found cluster centers. New centers\n"\
"    are appended to this list.\n"\
"dmin : float\n"\
"    Distance parameter for regular spatial clustering. Whenever a frame\n"\
"    is at least `dmin` away form all cluster centers it is added as a\n"\
"    new cluster center.\n"\
"metric : string\n"\
"    One of \"euclidian\" or \"minRMSD\" (case sensitive).\n"\
"max_clusters : unsigned integer\n"\
"    Maximum allowed number of cluster. The function will abort when\n"\
"    this limit is exceeded.\n"\
"\n"\
"Returns\n"\
"-------\n"\
"None. The input parameter `centers` is updated instead.\n"\
"\n"\
"Note\n"\
"----\n"\
"This function uses the minRMSD implementation of mdtraj."

static PyMethodDef regspatialMethods[] =
{
     {"cluster", cluster, METH_VARARGS, CLUSTER_USAGE},
     {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC
initregspatial(void)
{
  (void)Py_InitModule3("regspatial", regspatialMethods, MOD_USAGE);
  import_array();
}
