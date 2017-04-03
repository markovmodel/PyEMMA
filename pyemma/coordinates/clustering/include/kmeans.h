//
// Created by marscher on 4/3/17.
//

#ifndef PYEMMA_KMEANS_H
#define PYEMMA_KMEANS_H

#include <limits>
#include <ctime>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <clustering.h>
#include <metric.h>

namespace py = pybind11;


class KMeans {
public:
    KMeans(int k) : k(k) {

    }
protected:
    int k;
};


namespace {

//FIXME: this breakes the module's thread safety.
py::function set_callback;

void c_set_callback(py::function callback) {
    set_callback = callback;
}

}


template<typename dtype, typename metric_t>
py::list cluster(py::array_t<dtype, py::array::c_style> np_chunk,
                           py::list py_centers) {
    int debug;
    std::size_t N_centers, N_frames, dim;


    dtype *chunk;
    dtype **centers;
    dtype mindist;
    dtype d;
    int l;
    std::vector<int> centers_counter;
    void *arr_data;
    std::vector<dtype> new_centers;
    size_t i, j;
    int closest_center_index;
    debug = 0;
    closest_center_index = 0;
    if(debug) printf("KMEANS: \n----------- cluster called ----------\n");
    if(debug) printf("KMEANS: declaring variables...");
    if(debug) printf("done.\n");

    if(debug) printf("KMEANS: initializing some of them...");
    centers = NULL; chunk = NULL;
    if(debug) printf("done\n");

    if(np_chunk.ndim() != 2) {throw std::runtime_error("Number of dimensions of \"chunk\" isn\'t 2."); }
    N_frames = np_chunk.shape(0);
    dim = np_chunk.shape(1);
    if(dim==0) {
        throw std::runtime_error("chunk dimension must be larger than zero.");
    }
    metric_t metric(dim);

    chunk = np_chunk.mutable_data();
    if(debug) printf("done with N_frames=%zd, dim=%zd\n", N_frames, dim);

    /* import list of cluster centers */
    if(debug) printf("KMEANS: importing list of cluster centers...");
    N_centers = py_centers.size();
    if(!(centers = (float**) malloc(N_centers*sizeof(dtype)))) {
        throw std::runtime_error("could not allocate memory for centers");
    }
    for(size_t i = 0; i < N_centers; ++i) {
        //l = 0;
        py::array_t<dtype, py::array::c_style> np_item;

        if(debug) printf("%d", l++); /* 0 */
        // TODO: handle casting exceptions
        PyObject* py_item = py_centers[i].ptr(); //PyList_GetItem(py_centers,i); /* ref:borr. */
        np_item = py::cast(py_centers[i].ptr());
//        if(debug) printf("%d", l++); /* 1 */
//        if(debug) printf("%d", l++); /* 2 */
//        if(!PyArray_Check(py_item)) { PyErr_SetString(PyExc_ValueError, "Elements of centers must be numpy arrays."); goto error; }
//        if(debug) printf("%d", l++); /* 3 */
//        np_item = (PyArrayObject*)py_item;
//        if(debug) printf("%d", l++); /* 4 */
//        if(PyArray_TYPE(np_item)!=NPY_FLOAT32) { PyErr_SetString(PyExc_ValueError, "dtype of cluster center isn\'t float (32)."); goto error; };
//        if(debug) printf("%d", l++); /* 5 */
//        if(!PyArray_ISBEHAVED_RO(np_item) ) { PyErr_SetString(PyExc_ValueError, "cluster center isn\'t behaved."); goto error; };
//        if(debug) printf("%d", l++); /* 6 */
//        if(PyArray_NDIM(np_item)!=1) { PyErr_SetString(PyExc_ValueError, "Number of dimensions of cluster centers must be 1."); goto error;  };
//        if(debug) printf("%d", l++); /* 7 */
        if(np_item.shape(0) != dim) {
            throw std::runtime_error( "Dimension of cluster centers doesn\'t match dimension of frames.");
        }
//        if(debug) printf("%d", l++); /* 8 */
        centers[i] = (dtype*)PyArray_DATA(np_item.mutable_data());
        //if(debug) printf("%d", l++); /* 9 */
    }


    if(debug) printf("done, k=%zd\n", N_centers);
////////////////////////7 args////////////////7
    /* initialize centers_counter and new_centers with zeros */
    if(debug) printf("KMEANS: initializing calloc centers counter and new_centers stuff...");
    //centers_counter = (int*) calloc(N_centers, sizeof(int));
    centers_counter = std::vector<int>(N_centers, 0);
    //new_centers = (dtype*) calloc(N_centers * dim, sizeof(dtype));
    new_centers = std::vector<dtype>(N_centers * dim);

    /* do the clustering */
    if(debug) printf("KMEANS: performing the clustering...");
    int* centers_counter_p = centers_counter.data();
    dtype* new_centers_p = new_centers.data();
    for (i = 0; i < N_frames; i++) {
        mindist = std::numeric_limits<dtype>::max();
        for(j = 0; j < N_centers; ++j) {
            d = metric.compute(&chunk[i*dim], centers[j]);
            if(d<mindist) {
                mindist = d;
                closest_center_index = j;
            }
        }
        (*(centers_counter_p + closest_center_index))++;
        for (j = 0; j < dim; j++) {
            new_centers[closest_center_index*dim+j] += chunk[i*dim+j];
        }
    }

    for (i = 0; i < N_centers; i++) {
        if (*(centers_counter_p + i) == 0) {
            for (j = 0; j < dim; j++) {
                (*(new_centers_p + i * dim + j)) = centers[i][j];
            }
        } else {
            for (j = 0; j < dim; j++) {
                (*(new_centers_p + i * dim + j)) /= (*(centers_counter_p + i));
            }
        }
    }
    if(debug) printf("done\n");

    if(debug) printf("KMEANS: creating return_new_centers...");
    py::array_t<dtype> return_new_centers(N_centers*dim);
    //dims[0] = N_centers; dims[1] = dim;
    /*return_new_centers = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    if (return_new_centers == NULL){
        PyErr_SetString(PyExc_MemoryError, "Error occurs when creating a new PyArray");
        goto error;
    }
    arr_data = PyArray_DATA((PyArrayObject*)return_new_centers);
     */
    if(debug) printf("done\n");
    /* Need to copy the data of the malloced buffer to the PyObject
       since the malloced buffer will disappear after the C extension is called. */
    if(debug) printf("KMEANS: attempting memcopy...");
    memcpy(arr_data, new_centers.data(), return_new_centers.itemsize() * N_centers * dim);
    if(debug) printf("done\n");
    if(debug) printf("KMEANS: increasing ref to return_new_centers object...");
    // TODO: needed?
    //Py_INCREF(return_new_centers);  /* The returned list should still exist after calling the C extension */
    if(debug) printf("done\n");
    return return_new_centers;
}


template <typename dtype, typename metric_t>
void costFunction(py::array_t<dtype> np_data, py::list np_centers) {
    int i, r;
    dtype value, d;
    dtype *data, *centers;
    size_t dim, n_frames;

    value = 0.0;
    n_frames = np_data.shape(0);
    dim = np_data.shape(0);
    metric_t metric(dim);

    for(r = 0; r < np_centers.size(); r++) {
        // this is a list of numpy arrays.
        centers = (dtype*) PyArray_DATA(np_centers[r].ptr());
        for(i = 0; i < n_frames; i++) {
            //value += pow(distance(&data[i*dim], &centers[0], dim, buffer_a, buffer_b, NULL), 2);
            value += metric.compute(&data[i*dim], &centers[0]);
        }
    }
}


template <typename dtype, typename metric_t>
py::array_t<dtype, py::array::c_style>
initCentersKMpp(py::array_t<dtype, py::array::c_style> np_data, int k, bool use_random_seed) {
    size_t centers_found, first_center_index, n_trials;
    int some_not_done;
    dtype d;
    dtype dist_sum;
    dtype sum;
    size_t dim, n_frames;
    size_t i, j;
    int *taken_points;
    int best_candidate = -1;
    dtype best_potential = std::numeric_limits<dtype>::max();
    std::vector<int> next_center_candidates;
    std::vector<dtype> next_center_candidates_rand;
    std::vector<dtype> next_center_candidates_potential;
    dtype *data = nullptr;
    std::vector<dtype> init_centers, squared_distances;
    std::vector<dtype> arr_data;

    taken_points = NULL;
    centers_found = 0;
    dist_sum = 0.0;

    if(use_random_seed) {
#ifndef _KMEANS_INIT_RANDOM_SEED
#define _KMEANS_INIT_RANDOM_SEED
        /* set random seed */
        srand(time(NULL));
#endif
    } else {
#ifdef _KMEANS_INIT_RANDOM_SEED
#undef _KMEANS_INIT_RANDOM_SEED
#endif
        srand(42);
    }

    n_frames = np_data.shape(0);
    dim = np_data.shape(1);
    data = np_data.mutable_data(); //PyArray_DATA(np_data);
    /* number of trials before choosing the data point with the best potential */
    n_trials = 2 + (int) log(k);

    /* allocate space for the index giving away which point has already been used as a cluster center */
    if(!(taken_points = (int*) calloc(n_frames, sizeof(int)))) { PyErr_NoMemory();  }
    /* allocate space for the array holding the cluster centers to be returned */
    //if(!(init_centers = (float*) calloc(k * dim, sizeof(float)))) { PyErr_NoMemory();  }
    init_centers.reserve(k*dim);
    /* allocate space for the array holding the squared distances to the assigned cluster centers */
    //if(!(squared_distances = (float*) calloc(n_frames, sizeof(float)))) { PyErr_NoMemory(); }
    squared_distances.reserve(n_frames);

    /* candidates allocations */
    next_center_candidates.reserve(n_trials);
    next_center_candidates_rand.reserve(n_trials);
    next_center_candidates_potential.reserve(n_trials);
    metric_t metric(dim);

    /* pick first center randomly */
    first_center_index = rand() % n_frames;
    /* and mark it as assigned */
    taken_points[first_center_index] = 1;
    /* write its coordinates into the init_centers array */
    for(j = 0; j < dim; j++) {
        (*(init_centers.data() + centers_found*dim + j)) = data[first_center_index*dim + j];
    }
    /* increase number of found centers */
    centers_found++;
    /* perform callback */
    if(set_callback) {
        set_callback();
    }

    /* iterate over all data points j, measuring the squared distance between j and the initial center i: */
    /* squared_distances[i] = distance(x_j, x_i)*distance(x_j, x_i) */
    for(i = 0; i < n_frames; i++) {
        if(i != first_center_index) {
            d = pow(metric.compute(&data[i*dim], &data[first_center_index*dim]), 2);
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
                        d = pow(metric.compute(&data[i*dim], &data[next_center_candidates[j]*dim]), 2);
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
        best_potential = std::numeric_limits<dtype>::max();
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
                (*(init_centers.data() + centers_found*dim + j)) = (*(data + best_candidate*dim + j));
            }
            /* increase centers_found */
            centers_found++;
            /* perform the callback */
            if(set_callback) {
                set_callback();
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
                        d = pow(metric.compute(&data[i*dim], &data[best_candidate*dim]), 2);
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
    //dims[0] = k;
    //dims[1] = dim;
    //ret_init_centers = PyArray_SimpleNew(2, dims, NPY_FLOAT32);
    //if (ret_init_centers == NULL){
      //  PyErr_SetString(PyExc_MemoryError, "Error occurs when creating a new PyArray");
        //goto error;
    //}
    //arr_data = PyArray_DATA((PyArrayObject*)ret_init_centers);
    /* Need to copy the data of the malloced buffer to the PyObject
       since the malloced buffer will disappear after the C extension is called. */
    //memcpy(arr_data, init_centers, PyArray_ITEMSIZE((PyArrayObject*) ret_init_centers) * k * dim);
    //Py_INCREF(ret_init_centers);  /* The returned list should still exist after calling the C extension */


    /**
     *   explicit array_t(const std::vector<size_t> &shape, const T *ptr = nullptr,
            handle base = handle())
     */
     std::vector<size_t> shape;
    shape.push_back(k);
    shape.push_back(dim);
    py::array_t<dtype, py::array::c_style> ret_init_centers(shape);


    memcpy(ret_init_centers.mutable_data(), arr_data.data(), arr_data.size());
    return ret_init_centers;

    /*error:
    // reset the seed to something else than 42
    if(!use_random_seed) {
        srand(time(NULL));
    }
    free(taken_points);
    free(init_centers);
    free(squared_distances);
    free(next_center_candidates);
    free(next_center_candidates_rand);
    free(next_center_candidates_potential);
     */
    // return ret_init_centers;
}


#endif //PYEMMA_KMEANS_H
