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


import math
import os
from contextlib import contextmanager

import psutil
import random
import tempfile

from pyemma._base.progress.reporter import ProgressReporterMixin
from pyemma.coordinates.clustering.interface import AbstractClustering
from pyemma.util.annotators import fix_docs
from pyemma.util.units import bytes_to_string

from pyemma.util.contexts import random_seed, nullcontext

from deeptime.clustering import metrics, KMeans, MiniBatchKMeans

import numpy as np


__all__ = ['KmeansClustering', 'MiniBatchKmeansClustering']


@fix_docs
class KmeansClustering(AbstractClustering, ProgressReporterMixin):
    r"""k-means clustering"""

    __serialize_version = 0
    __serialize_fields = ('initial_centers_', '_converged', )

    def __init__(self, n_clusters, max_iter=5, metric='euclidean',
                 tolerance=1e-5, init_strategy='kmeans++', fixed_seed=False,
                 oom_strategy='memmap', stride=1, n_jobs=None, skip=0, clustercenters=None, keep_data=False):
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

            .. math:: C(S) = \sum_{i=1}^{k} \sum_{\mathbf x \in S_i} \left\| \mathbf x - \boldsymbol\mu_i \right\|^2

            is smaller than tolerance.
        metric : str
            metric to use during clustering ('euclidean', 'minRMSD')

        init_strategy : string
            can be either 'kmeans++' or 'uniform', determining how the initial
            cluster centers are being chosen

        fixed_seed : bool or int
            if True, the seed gets set to 42. Use time based seeding otherwise.
            if an integer is given, use this to initialize the random generator.

        oom_strategy : string, default='memmap'
            how to deal with out of memory situation during accumulation of all
            data.

            * 'memmap': if no memory is available to store all data, a memory
                mapped file is created and written to
            * 'raise': raise OutOfMemory exception.

        stride : int
            stridden data

        n_jobs : int or None, default None
            Number of threads to use during assignment of the data.
            If None, all available CPUs will be used.

        clustercenters: None or array(k, dim)
            This is used to resume the kmeans iteration. Note, that if this is set, the init_strategy is ignored and
            the centers are directly passed to the kmeans iteration algorithm.

        keep_data: boolean, default False
            If you intend to resume the kmeans iteration later on, in case it did not converge,
            this parameter controls whether the input data is kept in memory or not.

        """
        super(KmeansClustering, self).__init__(metric=metric, n_jobs=n_jobs)

        from ._ext import rmsd
        metrics.register("minRMSD", rmsd)

        if clustercenters is None:
            clustercenters = []

        self._in_memory_chunks_set = False
        self._converged = False

        self.set_params(n_clusters=n_clusters, max_iter=max_iter, tolerance=tolerance,
                        init_strategy=init_strategy, oom_strategy=oom_strategy,
                        fixed_seed=fixed_seed, stride=stride, skip=skip, clustercenters=clustercenters,
                        keep_data=keep_data
                        )

    @property
    def init_strategy(self):
        """Strategy to get an initial guess for the centers."""
        return self._init_strategy

    @init_strategy.setter
    def init_strategy(self, value):
        valid = ('kmeans++', 'uniform')
        if value not in valid:
            raise ValueError('invalid parameter "{}" for init_strategy. Should be one of {}'.format(value, valid))
        self._init_strategy = value

    @property
    def fixed_seed(self):
        """ seed for random choice of initial cluster centers. Fix this to get reproducible results."""
        return self._fixed_seed

    @fixed_seed.setter
    def fixed_seed(self, val):
        from pyemma.util import types
        if isinstance(val, bool) or val is None:
            if val:
                self._fixed_seed = 42
            else:
                self._fixed_seed = random.randint(0, 2**32-1)
        elif types.is_int(val):
            if val < 0 or val > 2**32-1:
                self.logger.warning("seed has to be positive (or smaller than 2**32-1)."
                                    " Seed will be chosen randomly.")
                self.fixed_seed = False
            else:
                self._fixed_seed = val
        else:
            raise ValueError("fixed seed has to be bool or integer")

        self.logger.debug("seed = %i", self._fixed_seed)

    @property
    def converged(self):
        return self._converged

    def _init_in_memory_chunks(self, size):
        # check if we need to allocate memory.
        if hasattr(self, '_in_memory_chunks') and self._in_memory_chunks.size == size:
            assert hasattr(self, '_in_memory_chunks')
            self.logger.debug("re-use in memory data.")
            return
        elif self._check_resume_iteration() and not self._in_memory_chunks_set and not self.keep_data:
            self.logger.warning('Resuming kmeans iteration without the setting "keep_data=True", will re-create'
                                ' the linear in-memory data. This is inefficient! Consider setting keep_data=True,'
                                ' when you intend to resume the kmeans iteration.')

        available_mem = psutil.virtual_memory().available
        required_mem = size * self.data_producer.dimension() * np.float32().itemsize
        if required_mem <= available_mem:
            self._in_memory_chunks = np.empty(shape=(size, self.data_producer.dimension()),
                                              order='C', dtype=np.float32)
        else:
            if self.oom_strategy == 'raise':
                self.logger.warning('K-means failed to load all the data (%s required, %s available) into memory. '
                                    'Consider using a larger stride or set the oom_strategy to \'memmap\' which works '
                                    'with a memmapped temporary file.'
                                    % (bytes_to_string(required_mem), bytes_to_string(available_mem)))
                raise MemoryError()
            else:
                self.logger.warning('K-means failed to load all the data (%s required, %s available) into memory '
                                    'and now uses a memmapped temporary file which is comparably slow. '
                                    'Consider using a larger stride.'
                                    % (bytes_to_string(required_mem), bytes_to_string(available_mem)))
                self._in_memory_chunks = np.memmap(tempfile.mkstemp()[1], mode="w+",
                                                   shape=(size, self.data_producer.dimension()), order='C',
                                                   dtype=np.float32)

    def describe(self):
        return "[Kmeans, k=%i, inp_dim=%i]" % (self.n_clusters, self.data_producer.dimension())

    def _check_resume_iteration(self):
        # if we have centers set, we should continue to iterate these
        return self.clustercenters.size != 0

    def _estimate(self, iterable, **kw):
        self._init_estimate()
        estimator = KMeans(self.n_clusters, self.max_iter, self.metric, tolerance=self.tolerance,
                           init_strategy=self.init_strategy, fixed_seed=self.fixed_seed, n_jobs=self.n_jobs)
        if len(self.clustercenters) > 0:
            estimator.initial_centers = np.array(self.clustercenters)
        ctx = nullcontext() if 'data' not in self._progress_registered_stages else self._progress_context(stage='data')
        # collect the data only if, we have not done this previously (eg. keep_data=True)
        # or the centers are not initialized.
        if not self._check_resume_iteration() or not self._in_memory_chunks_set:
            resume_centers = self._check_resume_iteration()
            with iterable.iterator(return_trajindex=True, stride=self.stride,
                                   chunk=self.chunksize, skip=self.skip) as it, ctx:
                # first pass: gather data and run k-means
                first_chunk = True
                for itraj, X in it:
                    # collect data
                    self._collect_data(X, first_chunk, it.last_chunk)
                    if not resume_centers:
                        # initialize cluster centers in case of uniform
                        self._initialize_centers(X, itraj, it.pos, it.last_chunk)
                    first_chunk = False
            self.initial_centers_ = self.clustercenters[:]

            self.logger.debug("Accumulated all data, running kmeans on %s", self._in_memory_chunks.shape)
            self._in_memory_chunks_set = True
        else:
            if len(self.clustercenters) != self.n_clusters:
                # TODO: this can be non-fatal, because the extension can handle it?!
                raise RuntimeError('Passed clustercenters do not match n_clusters: {} vs. {}'.
                                   format(len(self.clustercenters), self.n_clusters))

        if self.show_progress:
            if self.init_strategy == 'kmeans++':
                self._progress_register(self.n_clusters,
                                        description="initialize kmeans++ centers", stage=0)
            self._progress_register(self.max_iter, description="kmeans iterations", stage=1)

            callback_init = lambda: self._progress_update(1, stage=0)
            callback = lambda: self._progress_update(1, stage=1)
        else:
            callback_init = None
            callback = None

        with self._progress_context(stage=(0, 1)), self._finish_estimate():
            estimator.fit(self._in_memory_chunks, callback_init_centers=callback_init, callback_loop=callback,
                          n_jobs=self.n_jobs)
            model = estimator.fetch_model()
            self.clustercenters = model.cluster_centers
            self._converged = model.converged

        if self.converged:
            self.logger.debug("Cluster centers converged after %i steps.", len(model.inertias))
        else:
            self.logger.warning("Algorithm did not reach convergence criterion"
                                " of %g in %i iterations. Consider increasing max_iter.",
                                self.tolerance, self.max_iter)

        return self

    @contextmanager
    def _finish_estimate(self):
        try:
            yield
        finally:
            # delete the large input array, if the user wants to keep the array or the estimate has converged.
            if not self.keep_data or self._converged:
                fh = None
                if isinstance(self._in_memory_chunks, np.memmap):
                    fh = self._in_memory_chunks.filename
                del self._in_memory_chunks
                if fh:
                    os.unlink(fh)
                self._in_memory_chunks_set = False
            if self.init_strategy == 'uniform':
                del self._init_centers_indices

    def _init_estimate(self):
        # mini-batch sets stride to None
        stride = self.stride if self.stride else 1
        ###### init
        self._init_centers_indices = {}
        self._t_total = 0
        traj_lengths = self.trajectory_lengths(stride=stride, skip=self.skip)
        total_length = sum(traj_lengths)
        if not self.n_clusters:
            self.n_clusters = min(int(math.sqrt(total_length)), 5000)
            self.logger.info("The number of cluster centers was not specified, "
                             "using min(sqrt(N), 5000)=%s as n_clusters." % self.n_clusters)
        if not isinstance(self, MiniBatchKmeansClustering) and not self._source_from_memory(self.data_producer):
            n_chunks = self.data_producer.n_chunks(chunksize=self.chunksize, skip=self.skip, stride=self.stride)
            self._progress_register(n_chunks, description="creating data array", stage='data')

        self._init_in_memory_chunks(total_length)

        if self.init_strategy == 'uniform':
            # gives random samples from each trajectory such that the cluster centers are distributed percentage-wise
            # with respect to the trajectories length
            with random_seed(self.fixed_seed):
                for idx, traj_len in enumerate(traj_lengths):
                    self._init_centers_indices[idx] = random.sample(list(range(0, traj_len)), int(
                            math.ceil((traj_len / float(total_length)) * self.n_clusters)))

        return stride

    def _initialize_centers(self, X, itraj, t, last_chunk):
        if self.init_strategy == 'uniform':
            # needed for concatenation
            if len(self.clustercenters) == 0:
                self.clustercenters = np.empty((0, X.shape[1]))

            if itraj in list(self._init_centers_indices.keys()):
                for l in range(len(X)):
                    if len(self.clustercenters) < self.n_clusters and t + l in self._init_centers_indices[itraj]:
                        new = np.vstack((self.clustercenters, X[l]))
                        self.clustercenters = new

    def _collect_data(self, X, first_chunk, last_chunk):
        # beginning - compute
        if first_chunk:
            self._t_total = 0

        # appends a true copy
        self._in_memory_chunks[self._t_total:self._t_total + len(X)] = X[:]
        self._t_total += len(X)
        if 'data' in self._progress_registered_stages:
            self._progress_update(1, stage='data')

        if last_chunk:
            self._in_memory_chunks_set = True


class MiniBatchKmeansClustering(KmeansClustering):
    r"""Mini-batch k-means clustering"""

    __serialize_version = 0

    def __init__(self, n_clusters, max_iter=5, metric='euclidean', tolerance=1e-5, init_strategy='kmeans++',
                 batch_size=0.2, oom_strategy='memmap', fixed_seed=False, stride=None, n_jobs=None, skip=0,
                 clustercenters=None, keep_data=False):
        from ._ext import rmsd
        metrics.register("minRMSD", rmsd)

        if stride is not None:
            raise ValueError("stride is a dummy value in MiniBatch Kmeans")
        if batch_size > 1:
            raise ValueError("batch_size should be less or equal to 1, but was %s" % batch_size)
        if keep_data:
            raise ValueError("keep_data is a dummy value in MiniBatch Kmeans")

        super(MiniBatchKmeansClustering, self).__init__(n_clusters, max_iter, metric,
                                                        tolerance, init_strategy, False,
                                                        oom_strategy, stride=stride, n_jobs=n_jobs, skip=skip,
                                                        clustercenters=clustercenters, keep_data=False)

        self.set_params(batch_size=batch_size)

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

    def _init_estimate(self):
        self._traj_lengths = self.trajectory_lengths(skip=self.skip)
        self._total_length = sum(self._traj_lengths)
        samples = int(math.ceil(self._total_length * self.batch_size))
        self._n_samples = 0
        self._n_samples_traj = {}
        for idx, traj_len in enumerate(self._traj_lengths):
            traj_samples = int(math.floor(traj_len / float(self._total_length) * samples))
            self._n_samples_traj[idx] = traj_samples
            self._n_samples += traj_samples

        self._random_access_stride = np.empty(shape=(self._n_samples, 2), dtype=int)
        super(MiniBatchKmeansClustering, self)._init_estimate()

    def _estimate(self, iterable, **kw):
        # mini-batch kmeans does not use stride. Enforce it.

        estimator = MiniBatchKMeans(n_clusters=self.n_clusters, metric=self.metric, n_jobs=self.n_jobs,
                                    init_strategy=self.init_strategy)
        if len(self.clustercenters) > 0:
            estimator.initial_centers = np.array(self.clustercenters)

        self.stride = None
        self._init_estimate()

        i_pass = 0
        prev_cost = 0

        ra_stride = self._draw_mini_batch_sample()
        with iterable.iterator(return_trajindex=False, stride=ra_stride, skip=self.skip) as iterator, \
            self._progress_context(), self._finish_estimate():
            while not (self._converged or i_pass + 1 > self.max_iter):
                first_chunk = True
                # draw new sample and re-use existing iterator instance.
                ra_stride = self._draw_mini_batch_sample()
                iterator.stride = ra_stride
                iterator.reset()
                for X in iter(iterator):
                    # collect data
                    self._collect_data(X, first_chunk, iterator.last_chunk)
                    first_chunk = False

                current_model = estimator.partial_fit(self._in_memory_chunks).fetch_model()
                self.clustercenters = current_model.cluster_centers
                cost = current_model.inertia

                rel_change = np.abs(cost - prev_cost) / cost if cost != 0.0 else 0.0
                prev_cost = cost

                if rel_change <= self.tolerance:
                    self._converged = True
                    self.logger.info("Cluster centers converged after %i steps.", i_pass + 1)
                    self._progress_force_finish(stage=1)
                else:
                    self._progress_update(1, stage=1)

                i_pass += 1

        if not self._converged:
            self.logger.info("Algorithm did not reach convergence criterion"
                             " of %g in %i iterations. Consider increasing max_iter.", self.tolerance, self.max_iter)
        return self
