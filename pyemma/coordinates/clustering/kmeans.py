# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Created on 22.01.2015

@author: clonker, marscher, noe
"""

from __future__ import absolute_import

import math
import os
import psutil
import random
import tempfile

from pyemma._base.progress.reporter import ProgressReporter
from pyemma.coordinates.clustering.interface import AbstractClustering
from pyemma.util.annotators import doc_inherit
from pyemma.util.units import bytes_to_string

from six.moves import range
import numpy as np

from . import kmeans_clustering


__all__ = ['KmeansClustering']


class KmeansClustering(AbstractClustering, ProgressReporter):
    r"""k-means clustering"""

    def __init__(self, n_clusters, max_iter=5, metric='euclidean',
                 tolerance=1e-5, init_strategy='kmeans++', fixed_seed=False,
                 oom_strategy='memmap', stride=1):
        r"""Kmeans clustering

        Parameters
        ----------
        n_clusters : int
            amount of cluster centers. When not specified (None), min(sqrt(N), 5000) is chosen as default value,
            where N denotes the number of data points

        max_iter : int
            maximum number of iterations before stopping.

        tolerance : float
            stop iteration when the relative change in the cost function

            ..1:                C(S) = \sum_{i=1}^{k} \sum_{\mathbf x \in S_i} \left\| \mathbf x - \boldsymbol\mu_i \right\|^2

            is smaller than tolerance.
        metric : str
            metric to use during clustering ('euclidean', 'minRMSD')

        init_strategy : string
            can be either 'kmeans++' or 'uniform', determining how the initial
            cluster centers are being chosen

        fixed_seed : bool
            if True, the seed gets set to 42

        oom_strategy : string, default='memmap'
            how to deal with out of memory situation during accumulation of all
            data.

            * 'memmap': if no memory is available to store all data, a memory
                mapped file is created and written to
            * 'raise': raise OutOfMemory exception.

        stride : int
            stridden data

        """
        super(KmeansClustering, self).__init__(metric=metric)

        self.set_params(n_clusters=n_clusters, max_iter=max_iter, tolerance=tolerance,
                        init_strategy=init_strategy, oom_strategy=oom_strategy,
                        fixed_seed=fixed_seed, stride=stride,
                        )

        self._cluster_centers_iter = None
        self._centers_iter_list = []

    def _param_init(self):
        self._prev_cost = 0
        self._cluster_centers_iter = []
        self._init_centers_indices = {}
        self._t_total = 0
        traj_lengths = self.trajectory_lengths(stride=self._param_with_stride)
        total_length = sum(traj_lengths)

        if not self.n_clusters:
            self.n_clusters = min(int(math.sqrt(total_length)), 5000)
            self._logger.info("The number of cluster centers was not specified, "
                              "using min(sqrt(N), 5000)=%s as n_clusters." % self.n_clusters)

        if self.init_strategy == 'kmeans++':
            self._progress_register(self.n_clusters, description="initialize kmeans++ centers", stage=0)
        self._progress_register(self.max_iter, description="kmeans iterations", stage=1)

        self._init_in_memory_chunks(total_length)

        if self.init_strategy == 'uniform':
            # gives random samples from each trajectory such that the cluster centers are distributed percentage-wise
            # with respect to the trajectories length
            if self.fixed_seed:
                random.seed(42)
            for idx, traj_len in enumerate(traj_lengths):
                self._init_centers_indices[idx] = random.sample(list(range(0, traj_len)), int(
                    math.ceil((traj_len / float(total_length)) * self.n_clusters)))
            if self.fixed_seed:
                random.seed(None)

    def _init_in_memory_chunks(self, size):
        available_mem = psutil.virtual_memory().available
        required_mem = self._calculate_required_memory(size)
        if required_mem <= available_mem:
            self._in_memory_chunks = np.empty(shape=(size, self.data_producer.dimension()),
                                              order='C', dtype=np.float32)
        else:
            if self._oom_strategy == 'raise':
                self._logger.warning('K-means failed to load all the data (%s required, %s available) into memory. '
                                  'Consider using a larger stride or set the oom_strategy to \'memmap\' which works '
                                  'with a memmapped temporary file.'
                                  % (bytes_to_string(required_mem), bytes_to_string(available_mem)))
                raise MemoryError
            else:
                self._logger.warning('K-means failed to load all the data (%s required, %s available) into memory '
                                  'and now uses a memmapped temporary file which is comparably slow. '
                                  'Consider using a larger stride.'
                                  % (bytes_to_string(required_mem), bytes_to_string(available_mem)))
                self._in_memory_chunks = np.memmap(tempfile.mkstemp()[1], mode="w+",
                                                   shape=(size, self.data_producer.dimension()), order='C',
                                                   dtype=np.float32)

    def _calculate_required_memory(self, size):
        empty = np.empty(shape=(1, self.data_producer.dimension()), order='C', dtype=np.float32)
        return empty[0, :].nbytes * size

    @doc_inherit
    def describe(self):
        return "[Kmeans, k=%i, inp_dim=%i]" % (self.n_clusters, self.data_producer.dimension())

    def _param_finish(self):
        self.clustercenters = np.array(self._cluster_centers_iter)
        del self._cluster_centers_iter

        fh = None
        if isinstance(self._in_memory_chunks, np.memmap):
            fh = self._in_memory_chunks.filename
        del self._in_memory_chunks
        if fh:
            os.unlink(fh)

        if self.init_strategy == 'uniform':
            del self._centers_iter_list
            del self._init_centers_indices
        if self.init_strategy == 'kmeans++':
            self._progress_force_finish(0)
        self._progress_force_finish(1)

    def kmeanspp_center_assigned(self):
        self._progress_update(1, stage=0)

    def _estimate(self, iterable, **kw):

        stride = kw['stride'] if 'stride' in kw else self.stride

        iterator = iterable.iterator(return_trajindex=True, **kw)
        # first pass: gather data and run k-means
        first_chunk = True
        t = 0
        last_itraj = -1
        last_chunk = True if iterator.chunksize == 0 else False
        n_frames = iterator.n_frames_total()
        last_traj_len = iterator.trajectory_lengths()[-1]
        ntraj = self.number_of_trajectories()

        for itraj, X in iterator:
            if itraj != last_itraj:
                t = 0
                last_itraj = itraj
            t += len(X)

            if itraj == ntraj - 1:
                if iterator.chunksize == 0:
                    last_chunk = True
                elif t >= last_traj_len - iterator.chunksize:
                    last_chunk = True

            # collect data
            self._collect_data(X, first_chunk, stride)
            # initialize cluster centers
            self._logger.debug("last_chunk: %s" % last_chunk)
            self._initialize_centers(X, itraj, t, last_chunk)
            first_chunk = False

        # run k-means with all the data
        self._logger.debug("Accumulated all data, running kmeans on " + str(self._in_memory_chunks.shape))
        it = 0
        converged_in_max_iter = False
        while it < self.max_iter:
            self._cluster_centers_iter = kmeans_clustering.cluster(
                                                self._in_memory_chunks,
                                                self._cluster_centers_iter,
                                                self.metric)
            self._cluster_centers_iter = [row for row in self._cluster_centers_iter]

            cost = kmeans_clustering.cost_function(self._in_memory_chunks,
                                                   self._cluster_centers_iter,
                                                   self.metric,
                                                   self.n_clusters)
            rel_change = np.abs(cost - self._prev_cost) / cost
            self._prev_cost = cost

            if rel_change <= self.tolerance:
                converged_in_max_iter = True
                self._logger.info("Cluster centers converged after %i steps."
                                  % (it + 1))
                self._progress_force_finish(stage=1)
                break
            else:
                self._progress_update(1, stage=1)
            it += 1
        if not converged_in_max_iter:
            self._logger.info("Algorithm did not reach convergence criterion"
                              " of %g in %i iterations. Consider increasing max_iter."
                              % (self.tolerance, self.max_iter))

        return self

    def _initialize_centers(self, X, itraj, t, last_chunk):
        if self.init_strategy == 'uniform':
            if itraj in list(self._init_centers_indices.keys()):
                for l in range(len(X)):
                    if len(self._cluster_centers_iter) < self.n_clusters and t + l in self._init_centers_indices[itraj]:
                        self._cluster_centers_iter.append(X[l].astype(np.float32, order='C'))
        elif last_chunk and self.init_strategy == 'kmeans++':
            kmeans_clustering.set_callback(self.kmeanspp_center_assigned)
            cc = kmeans_clustering.init_centers(self._in_memory_chunks,
                                                self.metric, self.n_clusters, not self.fixed_seed)
            self._cluster_centers_iter = [c for c in cc]

    def _collect_data(self, X, first_chunk, stride):
        # beginning - compute
        if first_chunk:
            self._t_total = 0

        # appends a true copy
        self._in_memory_chunks[self._t_total:self._t_total + len(X)] = X[:]
        self._t_total += len(X)


class MiniBatchKmeansClustering(KmeansClustering):
    r"""Mini-batch k-means clustering"""

    def __init__(self, n_clusters, max_iter=5, metric='euclidean', tolerance=1e-5, init_strategy='kmeans++',
                 batch_size=0.2, oom_strategy='memmap', fixed_seed=False, stride=None):

        if stride is not None:
            raise ValueError("this is actually a dummy value... sorry")
        if batch_size > 1:
            raise ValueError("batch_size should be less or equal to 1, but was %s" % batch_size)

        self._cluster_centers_iter = None
        self._centers_iter_list = []

        super(MiniBatchKmeansClustering, self).__init__(n_clusters, max_iter, metric,
                                                        tolerance, init_strategy, False,
                                                        oom_strategy, stride=stride)

        self.set_params(batch_size=batch_size)

        self._param_with_stride = 1

    def _init_in_memory_chunks(self, size):
        return super(MiniBatchKmeansClustering, self)._init_in_memory_chunks(self._n_samples)

    def _draw_mini_batch_sample(self):
        offset = 0
        for idx, traj_len in enumerate(self._traj_lengths):
            n_samples_traj = self._n_samples_traj[idx]
            start = slice(offset, offset + n_samples_traj)

            self._random_access_stride[start, 0] = idx * np.ones(
                n_samples_traj, dtype=int)

            # draw 'n_samples_traj' without replacement from range(0, traj_len)
            choice = np.random.choice(traj_len, n_samples_traj, replace=False)

            self._random_access_stride[start, 1] = np.sort(choice).T
            offset += n_samples_traj

        return self._random_access_stride

    def _param_init(self):
        self._traj_lengths = self.trajectory_lengths(stride=self._param_with_stride)
        self._total_length = sum(self._traj_lengths)
        samples = int(math.ceil(self._total_length * self.batch_size))
        self._n_samples = 0
        self._n_samples_traj = {}
        self._prev_cost = 0
        for idx, traj_len in enumerate(self._traj_lengths):
            traj_samples = int(math.floor(traj_len / float(self._total_length) * samples))
            self._n_samples_traj[idx] = traj_samples
            self._n_samples += traj_samples

        self._random_access_stride = np.empty(shape=(self._n_samples, 2), dtype=int)

        super(MiniBatchKmeansClustering, self)._param_init()

    def _estimate(self, iterable, **kw):
        if 'stride' in kw:
            raise ValueError("no stride parameter allowed for minibatch kmeans.")
        ntraj = self.number_of_trajectories()

        ipass = 0
        converged_in_max_iter = False

        ra_stride = self._draw_mini_batch_sample()
        iterator = iterable.iterator(return_trajindex=True, stride=ra_stride)

        while not (converged_in_max_iter or ipass + 1 >= self.max_iter):
            # TODO: move first_chunk etc. to iterable/datasource?
            first_chunk = True
            last_chunk = False
            last_itraj = -1
            t = 0
            # draw new sample and re-use existing iterator instance.
            ra_stride = self._draw_mini_batch_sample()
            iterator.stride = ra_stride
            iterator.reset()
            for itraj, X in iter(iterator):
                if last_itraj != itraj:
                    last_itraj = itraj
                    t = 0
                L = len(X)

                # last chunk in traj?
                traj_len = iterator.ra_trajectory_length(itraj)
                last_chunk_in_traj = t + L >= traj_len
                # last chunk?
                last_chunk = last_chunk_in_traj and itraj >= ntraj - 1
                # collect data
                self._collect_data(X, first_chunk, None)
                # initialize cluster centers
                if ipass == 0:
                    self._initialize_centers(X, itraj, t, last_chunk)
                first_chunk = False
                t += L
            # one pass over data completed
            self._cluster_centers_iter = kmeans_clustering.cluster(self._in_memory_chunks,
                                                                   self._cluster_centers_iter,
                                                                   self.metric)
            self._cluster_centers_iter = [row for row in self._cluster_centers_iter]

            cost = kmeans_clustering.cost_function(self._in_memory_chunks,
                                                   self._cluster_centers_iter,
                                                   self.metric,
                                                   self.n_clusters)

            rel_change = np.abs(cost - self._prev_cost) / cost
            self._prev_cost = cost
            self._cluster_centers = np.array(self._cluster_centers_iter)

            if rel_change <= self.tolerance:
                converged_in_max_iter = True
                self._logger.info("Cluster centers converged after %i steps." % (ipass + 1))
                self._progress_force_finish(stage=1)
            else:
                self._progress_update(1, stage=1)

            ipass += 1

        if not converged_in_max_iter:
            self._logger.info("Algorithm did not reach convergence criterion"
                              " of %g in %i iterations. Consider increasing max_iter."
                              % (self.tolerance, self.max_iter))

        return self
