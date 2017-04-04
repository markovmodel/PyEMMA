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
#include <metric.h>
#include <kmeans.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;


static char MOD_USAGE[] = "kmeans clustering";

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
"\n"\
"Returns\n"\
"-------\n"\
"None. The input parameter `centers` is updated instead.\n"\
"\n"\
"Note\n"\
"----\n"\
"This function uses the minRMSD implementation of mdtraj.";

static char INIT_CENTERS_USAGE[] = "init_centers(data, metric, k)\n"\
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
"use_random_seed : bool\n"\
"    (input) determines if a fixed seed should be used or a random one\n"\
"\n"\
"Returns\n"\
"-------\n"\
"A numpy ndarray of cluster centers assigned to the provided data set.\n"\
"\n"\
"Note\n"\
"----\n"\
"This function uses the minRMSD implementation of mdtraj.";


PYBIND11_PLUGIN(kmeans_clustering) {
        py::module m("kmeans_clustering", MOD_USAGE);

       // typedef metric::min_rmsd_metric minRMSD_f;
        typedef metric::euclidean_metric<float> euclidean_f;

        //m.def("cluster_minRMSD_f", &cluster<float, minRMSD_f>, CLUSTER_USAGE);
        m.def("cluster_euclidean_f", &cluster<float, euclidean_f>, CLUSTER_USAGE);

        //m.def("cluster_minRMSD_d", &cluster<double, min>, CLUSTER_USAGE);
        //m.def("cluster_euclidean_d", &cluster<double, metric::euclidean_metric>, CLUSTER_USAGE);

        m.def("assign", &assign, ASSIGN_USAGE);
       // m.def("init_centers_minRMSD_f", &initCentersKMpp<float, minRMSD_f>, INIT_CENTERS_USAGE);
       // m.def("cost_function_minRMSD_f", &costFunction<float, minRMSD_f>, "Evaluates the cost function for the k-means clustering algorithm.");

        m.def("init_centers_euclidean_f", &initCentersKMpp<float, euclidean_f>);
        m.def("cost_function_euclidean_f", &costFunction<float, euclidean_f> );

        m.def("set_callback", &c_set_callback, "For setting a callback.");

        return m.ptr();
}