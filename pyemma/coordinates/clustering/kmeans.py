# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
Created on 22.01.2015

@author: marscher, noe
"""
import math
import os
import random
import tempfile
import numpy as np

from . import kmeans_clustering

from pyemma.util.annotators import doc_inherit
from pyemma.util.progressbar import ProgressBar
from pyemma.util.progressbar.gui import show_progressbar
from pyemma.coordinates.clustering.interface import AbstractClustering

__all__ = ['KmeansClustering']


class KmeansClustering(AbstractClustering):

    def __init__(self, n_clusters, max_iter=5, metric='euclidean',
                 tolerance=1e-5, init_strategy='kmeans++', oom_strategy='memmap'):
        r"""
        Kmeans clustering

        Parameters
        ----------
        n_clusters : int
            amount of cluster centers
        max_iter : int
            how many iterations per chunk?
        metric : str
            metric to use during clustering ('euclidean', 'minRMSD')
        tolerance : float
            if the cluster centers' change did not exceed tolerance, stop iterating
        init_strategy : string
            can be either 'kmeans++' or 'uniform', determining how the initial cluster centers are being chosen
        oom_strategy : string
            how to deal with out of memory situation during accumulation of all data.
            Currently if no memory is available to store all data, a memory mapped
            file is created and written to, if set to 'memmap'.
            Set it to 'raise', to raise the exception then.
        """
        super(KmeansClustering, self).__init__(metric=metric)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self._cluster_centers = []
        self._centers_iter_list = []
        self._tolerance = tolerance
        self._init_strategy = init_strategy
        self._oom_strategy = oom_strategy
        self._custom_param_progress_handling = True

    def _param_init(self):
        self._cluster_centers = []
        self._init_centers_indices = {}
        self._t_total = 0
        if self._init_strategy == 'kmeans++':
            self._progress_init = ProgressBar(self.n_clusters, description="initialize kmeans++ centers")
        self._progress_iters = ProgressBar(self.max_iter, description="kmeans iterations")
        traj_lengths = self.trajectory_lengths(stride=self._param_with_stride)
        total_length = sum(traj_lengths)
        try:
            self._in_memory_chunks = np.empty(shape=(total_length, self.data_producer.dimension()),
                                              order='C', dtype=np.float32)
        except MemoryError:
            if self._oom_strategy == 'raise':
                raise
            self._in_memory_chunks = np.memmap(tempfile.mkstemp()[1], mode="w+",
                                               shape=(total_length, self.data_producer.dimension()), order='C',
                                               dtype=np.float32)

        if self._init_strategy == 'uniform':
            # gives random samples from each trajectory such that the cluster centers are distributed percentage-wise
            # with respect to the trajectories length
            for idx, traj_len in enumerate(traj_lengths):
                self._init_centers_indices[idx] = random.sample(range(0, traj_len), int(
                    math.ceil((traj_len / float(total_length)) * self.n_clusters)))

    @doc_inherit
    def describe(self):
        return "[Kmeans, k=%i]" % self.n_clusters

    def _param_finish(self):
        self.clustercenters = np.array(self._cluster_centers)
        del self._cluster_centers

        fh = None
        if isinstance(self._in_memory_chunks, np.memmap):
            fh = self._in_memory_chunks.filename
        del self._in_memory_chunks
        if fh:
            os.unlink(fh)

        if self._init_strategy == 'uniform':
            del self._centers_iter_list
            del self._init_centers_indices
        self._progress_init.numerator = self._progress_init.denominator
        self._progress_init._eta.eta_epoch = 0
        self._progress_iters.numerator = self._progress_iters.denominator
        self._progress_iters._eta.eta_epoch = 0
        show_progressbar(self._progress_init)
        show_progressbar(self._progress_iters)

    def kmeanspp_center_assigned(self):
        self._progress_init.numerator += 1
        show_progressbar(self._progress_init)

    def _param_add_data(self, X, itraj, t, first_chunk, last_chunk_in_traj, last_chunk, ipass, Y=None, stride=1):
        # first pass: gather data and run k-means
        if ipass == 0:
            # beginning - compute
            if first_chunk:
                mem_req = int(1.0/1024**2 * X[0, :].nbytes * self.n_frames_total(stride))
                if mem_req > 200:
                    self._logger.warn('K-means implementation is currently memory inefficient.'
                                      ' This calculation needs %i megabytes of main memory.'
                                      ' If you get a memory error, try using a larger stride.'
                                      % mem_req)

            # appends a true copy
            self._in_memory_chunks[self._t_total:self._t_total + len(X)] = X[:]
            self._t_total += len(X)
            # initialize uniform cluster centers
            if self._init_strategy == 'uniform':
                if itraj in self._init_centers_indices.keys():
                    for l in xrange(len(X)):
                        if len(self._cluster_centers) < self.n_clusters and t + l in self._init_centers_indices[itraj]:
                            self._cluster_centers.append(X[l].astype(np.float32, order='C'))

            # run k-means in the end
            if last_chunk:
                # free part of the memory

                if self._init_strategy == 'kmeans++':
                    kmeans_clustering.set_callback(self.kmeanspp_center_assigned)
                    cc = kmeans_clustering.init_centers(self._in_memory_chunks,
                                                        self.metric, self.n_clusters)
                    self._cluster_centers = [c for c in cc]
                # run k-means with all the data
                self._logger.debug("Accumulated all data, running kmeans on " + str(self._in_memory_chunks.shape))
                it = 0
                converged_in_max_iter = False
                while it < self.max_iter:
                    # self._logger.info("step %i" % (it + 1))
                    old_centers = self._cluster_centers
                    self._cluster_centers = kmeans_clustering.cluster(self._in_memory_chunks,
                                                                      self._cluster_centers, self.metric)
                    self._cluster_centers = [row for row in self._cluster_centers]
                    if np.allclose(old_centers, self._cluster_centers, rtol=self._tolerance):
                        converged_in_max_iter = True
                        self._logger.info("Cluster centers converged after %i steps."
                                          % (it + 1))
                        self._progress_iters.numerator = self.max_iter
                        break
                    else:
                        self._progress_iters.numerator += 1
                    it += 1
                    show_progressbar(self._progress_iters)
                if not converged_in_max_iter:
                    self._logger.info("Algorithm did not reach convergence criterion"
                                      " of %g in %i iterations. Consider increasing max_iter."
                                      % (self._tolerance, self.max_iter))

            # done
            if last_chunk:
                return True
        return True
