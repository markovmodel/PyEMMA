//
// Created by marscher on 7/24/17.
//


#ifndef PYEMMA_KMEANS_BITS_H_H
#define PYEMMA_KMEANS_BITS_H_H

#include "kmeans.h"

template <typename dtype>
py::list KMeans<dtype>::cluster(const py::array_t<dtype, py::array::c_style>& np_chunk,
                 py::list py_centers) {
    int debug;

    size_t i, j;
    debug = 1;

    if(np_chunk.ndim() != 2) { throw std::runtime_error("Number of dimensions of \"chunk\" isn\'t 2."); }

    size_t N_frames = np_chunk.shape(0);
    size_t dim = np_chunk.shape(1);

    if(dim == 0) {
        throw std::runtime_error("chunk dimension must be larger than zero.");
    }

    auto chunk = np_chunk.unchecked<2>();
    if(debug) printf("done with N_frames=%zd, dim=%zd\n", N_frames, dim);

    /* import list of cluster centers */
    if(debug) printf("KMEANS: importing list of cluster centers...");
    size_t N_centers = py_centers.shape(0);

    // TODO: check centers should have shape (k, ndim)
    // FIXME: access pattern to centers is based on [][], but now its linear...
    dtype* centers = (dtype*)py_centers.mutable_data();

//    for(i = 0; i < N_centers; ++i) {
//        auto py_item = py_centers[i].ptr();
//        if (! py::isinstance<py::array>(py_item)) {
//            throw std::runtime_error("py_centers does not exclusively contain numpy arrays.");
//        }
//        auto np_item = py::reinterpret_borrow<py::array>(py_item);
//        //PyArrayObject* np_item = (PyArrayObject*)py_item;
//        if(np_item.dtype() != py::dtype("float")) { throw std::runtime_error("dtype of cluster center isn\'t float (32).");  };
//        //if(!PyArray_ISBEHAVED_RO(np_item) ) { throw std::runtime_error("cluster center isn\'t behaved.");  };
//        //if(PyArray_NDIM(np_item)!=1) { throw std::runtime_error("Number of dimensions of cluster centers must be 1.");   };
//        if (np_item.shape(1) != 1) { throw std::runtime_error("Number of dimensions of cluster centers must be 1."); }
//        if(np_item.shape(0) != dim) {
//            throw std::runtime_error("Dimension of cluster centers doesn\'t match dimension of frames.");
//        }
//        centers[i] = (float*)PyArray_DATA(np_item);
//    }

    if(debug) printf("done, k=%zd\n", N_centers);
    /* initialize centers_counter and new_centers with zeros */
    std::vector<int> centers_counter(N_centers, 0);
    std::vector<dtype> new_centers(N_centers * dim, 0.0);

    /* do the clustering */
    if(debug) printf("KMEANS: performing the clustering...");
    int* centers_counter_p = centers_counter.data();
    dtype* new_centers_p = new_centers.data();
    dtype mindist;
    size_t closest_center_index = 0;
    dtype d;
    for (i = 0; i < N_frames; i++) {
        mindist = std::numeric_limits<dtype>::max();
        for(j = 0; j < N_centers; ++j) {
            d = metric->compute(&chunk[i*dim], centers[j]);
            if(d < mindist) {
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
    void* arr_data = return_new_centers.mutable_data();
    if(debug) printf("done\n");
    /* Need to copy the data of the malloced buffer to the PyObject
       since the malloced buffer will disappear after the C extension is called. */
    if(debug) printf("KMEANS: attempting memcopy...");
    memcpy(arr_data, new_centers_p, return_new_centers.itemsize() * N_centers * dim);
    if(debug) printf("done\n");
    return return_new_centers;
}

// TODO: this could be private?
template <typename dtype>
dtype KMeans<dtype>::costFunction(py::array_t<dtype, py::array::c_style> np_data,
                   py::list np_centers) {
    int i, r;
    dtype value, d;
    dtype *data = np_data.data(), *centers;
    std::size_t dim, n_frames;

    value = 0.0;
    n_frames = np_data.shape(0);
    dim = np_data.shape(1);

    for (r = 0; r < np_centers.size(); r++) {
        // this is a list of numpy arrays.
        py::array_t <dtype> center_r = np_centers[r];
        centers = (dtype *) center_r.data();
        for (i = 0; i < n_frames; i++) {
            value += metric->compute(&data[i * dim], &centers[0]);
        }
    }
    return value;
}

template<typename  dtype>
py::array_t<dtype, py::array::c_style> Kmeans<dtype>::
        initCentersKMpp(py::array_t<dtype, py::array::c_style|py::array::forcecast> np_data, int k, bool use_random_seed) {
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
    if(callback) {
        callback();
    }

    /* iterate over all data points j, measuring the squared distance between j and the initial center i: */
    /* squared_distances[i] = distance(x_j, x_i)*distance(x_j, x_i) */
    for(i = 0; i < n_frames; i++) {
        if(i != first_center_index) {
            d = pow(metric->compute(&data[i*dim], &data[first_center_index*dim]), 2);
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
                        d = pow(metric->compute(&data[i*dim], &data[next_center_candidates[j]*dim]), 2);
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
                callback();
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
                        d = pow(metric->compute(&data[i*dim], &data[best_candidate*dim]), 2);
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
    std::vector<size_t> shape = {k, dim};
    py::array_t<dtype, py::array::c_style> ret_init_centers(shape);

    memcpy(ret_init_centers.mutable_data(), arr_data.data(), arr_data.size());
    return ret_init_centers;
}

#endif //PYEMMA_KMEANS_BITS_H_H
