//
// Created by marscher on 7/24/17.
//


#ifndef PYEMMA_KMEANS_BITS_H_H
#define PYEMMA_KMEANS_BITS_H_H

#include "kmeans.h"
#include <pybind11/pytypes.h>
#include <random>
#include<atomic>


#undef NDEBUG
#include <cassert>


template<typename dtype>
typename KMeans<dtype>::np_array
KMeans<dtype>::cluster(const np_array& np_chunk, const np_array& np_centers, int n_threads) const {
    using size_t = std::size_t;

    if (np_chunk.ndim() != 2) {
        throw std::runtime_error(R"(Number of dimensions of "chunk" ain't 2.)");
    }
    if (np_centers.ndim() != 2) {
        throw std::runtime_error(R"(Number of dimensions of "centers" ain't 2.)");
    }

    auto N_frames = static_cast<size_t>(np_chunk.shape(0));
    auto dim = static_cast<size_t>(np_chunk.shape(1));

    if (dim == 0) {
        throw std::invalid_argument("chunk dimension must be larger than zero.");
    }

    auto chunk = np_chunk.template unchecked<2>();
    auto N_centers = static_cast<size_t>(np_centers.shape(0));
    auto centers = np_centers.template unchecked<2>();

    std::vector<size_t> shape = {N_centers, dim};
    py::array_t <dtype> return_new_centers(shape);
    auto new_centers = return_new_centers.mutable_unchecked();
    std::fill(return_new_centers.mutable_data(), return_new_centers.mutable_data() + return_new_centers.size(), 0.0);

    /* initialize centers_counter and new_centers with zeros */
    std::vector<std::size_t> centers_counter(N_centers, 0);

    /* do the clustering */
#ifndef USE_OPENMP
    // dtype min = std::numeric_limits<dtype>::max();
    size_t closest_center_index = 0;
    for (i = 0; i < N_frames; i++) {
        dtype mindist = std::numeric_limits<dtype>::max();
        for(j = 0; j < N_centers; ++j) {
            auto d = parent_t::metric->compute(&chunk(i, 0), &centers(j, 0));
            if(d < mindist) {
                mindist = d;
                closest_center_index = j;
            }
        }
        centers_counter.at(closest_center_index)++;
        //(*(centers_counter_p + closest_center_index))++;
        for (j = 0; j < dim; j++) {
            new_centers(closest_center_index, j) += chunk(i, j);
        }
    }
#else
    omp_set_num_threads(n_threads);

    #pragma omp parallel for
    for (size_t i = 0; i < N_frames; ++i) {
        std::vector<dtype> dists(N_centers);
        for (size_t j = 0; j < N_centers; ++j) {
            dists[j] = parent_t::metric->compute(&chunk(i, 0), &centers(j, 0));
        }
        #pragma omp flush(dists)

        #pragma omp critical(centers_counter)
        {
            auto closest_center_index = std::distance(dists.begin(), std::min_element(dists.begin(), dists.end()));
            {
                centers_counter.at(static_cast<size_t>(closest_center_index))++;
                for (std::size_t j = 0; j < dim; j++) {
                    new_centers(closest_center_index, j) += chunk(i, j);
                }
            }
        }
    }
#endif
    auto centers_counter_it = centers_counter.begin();
    for (std::size_t i = 0; i < N_centers; ++i, ++centers_counter_it) {
        if (*centers_counter_it == 0) {
            for (std::size_t j = 0; j < dim; ++j) {
                new_centers(i, j) = centers(i, j);
            }
        } else {
            for (std::size_t j = 0; j < dim; ++j) {
                new_centers(i, j) /= static_cast<dtype>(*centers_counter_it);
            }
        }
    }
    return return_new_centers;
}

template<typename dtype>
dtype KMeans<dtype>::costFunction(const np_array& np_data, const np_array& np_centers, int n_threads) const {
    auto data = np_data.template unchecked<2>();
    auto centers = np_centers.template unchecked<2>();

    dtype value = 0.0;
    //std::atomic<dtype> value;
    //value = 0.0;
    std::size_t n_frames = np_data.shape(0);
    #ifdef USE_OPENMP
    omp_set_num_threads(n_threads);
    #endif

    #pragma omp parallel for reduction(+:value)
    for (size_t i = 0; i < n_frames; i++) {
        for (size_t r = 0; r < np_centers.shape(0); r++) {
            auto l = parent_t::metric->compute(&data(i, 0), &centers(r, 0));
            //#pragma omp critical
            {
                value = value + l;
            }
        }
    }
    return value;
}

template<typename dtype>
typename KMeans<dtype>::np_array KMeans<dtype>::
initCentersKMpp(const np_array& np_data, unsigned int random_seed, int n_threads) const {
    using size_t = std::size_t;

    constexpr auto size_t_max = std::numeric_limits<std::size_t>::max();

    size_t centers_found = 0, first_center_index;
    dtype sum;
    size_t i, j;
    size_t dim = parent_t::metric->dim;

    if (np_data.ndim() != 2) {
        throw std::invalid_argument("input data does not have two dimensions.");
    }

    if (np_data.shape(1) != dim) {
        throw std::invalid_argument("input dimension of data does not match the requested metric ones.");
    }

    auto n_frames = static_cast<size_t>(np_data.shape(0));

    /* number of trials before choosing the data point with the best potential */
    size_t n_trials = 2 + (size_t) log(k);

    /* allocate space for the index giving away which point has already been used as a cluster center */
    std::vector<bool> taken_points(n_frames, false);
    /* candidates allocations */

    std::vector<size_t> next_center_candidates(n_trials, size_t_max);
    std::vector<dtype> next_center_candidates_rand(n_trials, 0);
    std::vector<dtype> next_center_candidates_potential(n_trials, 0);
    /* allocate space for the array holding the squared distances to the assigned cluster centers */
    std::vector<dtype> squared_distances(n_frames, 0);

    /* create the output objects */
    std::vector<size_t> shape = {k, dim};
    np_array ret_init_centers(shape);
    auto init_centers = ret_init_centers.mutable_unchecked();
    std::memset(init_centers.mutable_data(), 0, init_centers.size() * init_centers.itemsize());

    auto data = np_data.template unchecked<2>();

    /* initialize random device and pick first center randomly */
    std::default_random_engine generator(random_seed);
    std::uniform_int_distribution<size_t> uniform_dist(0, n_frames - 1);
    first_center_index = uniform_dist(generator);
    /* and mark it as assigned */
    taken_points[first_center_index] = true;
    /* write its coordinates into the init_centers array */
    for (j = 0; j < dim; j++) {
        init_centers(centers_found, j) = data(first_center_index, j);
    }
    /* increase number of found centers */
    centers_found++;
    /* perform callback */
    if (!callback.is_none()) {
        callback();
    }
#ifdef USE_OPENMP
    omp_set_num_threads(n_threads);
#endif

    /* iterate over all data points j, measuring the squared distance between j and the initial center i: */
    /* squared_distances[i] = distance(x_j, x_i)*distance(x_j, x_i) */
    dtype dist_sum = 0.0;
    #pragma omp parallel for
    for (i = 0; i < n_frames; i++) {
        if (i != first_center_index) {
            auto value = parent_t::metric->compute(&data(i, 0), &data(first_center_index, 0));
            value = value * value;
            squared_distances[i] = value;
            /* build up dist_sum which keeps the sum of all squared distances */
            #pragma omp critical
            {
                dist_sum += value;
            }
        }
    }

    /* keep picking centers while we do not have enough of them... */
    while (centers_found < k) {

        /* initialize the trials random values by the D^2-weighted distribution */
        for (j = 0; j < n_trials; j++) {
            next_center_candidates[j] = size_t_max;
            auto point_index = uniform_dist(generator);
            next_center_candidates_rand[j] = dist_sum * (static_cast<dtype>(point_index) /
                                                         static_cast<dtype>(uniform_dist.max()));
            next_center_candidates_potential[j] = 0.0;
        }
        /* pick candidate data points corresponding to their random value */
        sum = 0.0;
        for (i = 0; i < n_frames; i++) {
            if (!taken_points[i]) {
                sum += squared_distances[i];
                bool some_not_done {false};
                for (j = 0; j < n_trials; j++) {
                    if (next_center_candidates[j] == size_t_max) {
                        if (sum >= next_center_candidates_rand[j]) {
                            assert(i < std::numeric_limits<int>::max());
                            next_center_candidates[j] = i;
                        } else {
                            some_not_done = true;
                        }
                    }
                }
                if (!some_not_done) break;
            }
        }

        {
            /* now find the maximum squared distance for each trial... */
            std::atomic_bool terminate(false);
            #pragma omp parallel
            for (size_t i = 0; i < n_frames; i++) {
                if(!terminate.load()) continue;
                if (!taken_points[i]) {
                    for (size_t j = 0; j < n_trials; ++j) {
                        if (next_center_candidates[j] == size_t_max) {
                            terminate.store(true);
                            break;
                        }
                        if (next_center_candidates[j] != i) {
                            auto value = parent_t::metric->compute(&data(i, 0), &data(next_center_candidates[j], 0));
                            auto d = value * value;
                            if (d < squared_distances[i]) {
                                next_center_candidates_potential[j] += d;
                            } else {
                                next_center_candidates_potential[j] += squared_distances[i];
                            }
                        }
                    }
                }
            }
        }

        /* ... and select the best candidate by the minimum value of the maximum squared distances */
        long best_candidate = -1;
        auto best_potential = std::numeric_limits<dtype>::max();
        for (j = 0; j < n_trials; j++) {
            if (next_center_candidates[j] != size_t_max && next_center_candidates_potential[j] < best_potential) {
                best_potential = next_center_candidates_potential[j];
                best_candidate = next_center_candidates[j];
            }
        }

        /* if for some reason we did not find a best candidate, just take the next available point */
        if (best_candidate == -1) {
            for (i = 0; i < n_frames; i++) {
                if (!taken_points[i]) {
                    best_candidate = i;
                    break;
                }
            }
        }

        /* check if best_candidate was set, otherwise break to avoid an infinite loop should things go wrong */
        if (best_candidate >= 0) {
            /* write the best_candidate's components into the init_centers array */
            for (j = 0; j < dim; j++) {
                init_centers(centers_found, j) = data(best_candidate, j);
            }
            /* increase centers_found */
            centers_found++;
            /* perform the callback */
            if (!callback.is_none()) {
                callback();
            }
            /* mark the data point as assigned center */
            taken_points[best_candidate] = true;
            /* update the sum of squared distances by removing the assigned center */
            dist_sum -= squared_distances[best_candidate];

            /* if we still have centers to assign, the squared distances array has to be updated */
            if (centers_found < k) {
                /* Check for each data point if its squared distance to the freshly added center is smaller than */
                /* the squared distance to the previously picked centers. If so, update the squared_distances */
                /* array by the new value and also update the dist_sum value by removing the old value and adding */
                /* the new one. */
                #pragma omp parallel for
                for (size_t i = 0; i < n_frames; ++i) {
                    if (!taken_points[i]) {
                        auto value = parent_t::metric->compute(&data(i, 0), &data(best_candidate, 0));
                        auto d = value * value;
                        #pragma omp critical
                        {
                            if (d < squared_distances[i]) {
                                dist_sum += d - squared_distances[i];
                                squared_distances[i] = d;
                            }
                        }
                    }
                }
            }
        } else {
            break;
        }
    }
    if (centers_found != k) { throw std::runtime_error("kmeans++ failed to initialize all desired centers"); }
    return ret_init_centers;
}

#endif //PYEMMA_KMEANS_BITS_H_H
