
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

'''
Created on 18.02.2015

@author: marscher
'''


import os

import numpy as np
from deeptime.clustering import ClusterModel, metrics

from pyemma._base.serialization.serialization import SerializableMixIn

from pyemma._base.model import Model
from pyemma._base.parallel import NJobsMixIn
from pyemma._ext.sklearn.base import ClusterMixin
from pyemma.coordinates.data._base.transformer import StreamingEstimationTransformer
from pyemma.util.annotators import fix_docs, aliased, alias
from pyemma.util.discrete_trajectories import index_states, sample_indexes_by_state
from pyemma.util.files import mkdir_p


@fix_docs
@aliased
class AbstractClustering(StreamingEstimationTransformer, Model, ClusterMixin, NJobsMixIn, SerializableMixIn):

    """
    provides a common interface for cluster algorithms.

    Parameters
    ----------

    metric: str, default='euclidean'
       metric to pass to c extension
    n_jobs: int or None, default=None
        How much threads to use during assignment
        If None, all available CPUs will be used.
    """

    def __init__(self, metric='euclidean', n_jobs=None):
        super(AbstractClustering, self).__init__()
        from ._ext import rmsd
        metrics.register("minRMSD", rmsd)
        self.metric = metric
        self.clustercenters = None
        self._previous_stride = -1
        self._dtrajs = []
        self._overwrite_dtrajs = False
        self._index_states = []
        self.n_jobs = n_jobs

    __serialize_fields = ('_dtrajs', '_previous_stride', '_index_states', '_overwrite_dtrajs', '_precentered')
    __serialize_version = 0

    def set_model_params(self, clustercenters):
        self.clustercenters = clustercenters

    @property
    @alias('cluster_centers_')  # sk-learn compat.
    def clustercenters(self):
        """ Array containing the coordinates of the calculated cluster centers. """
        return self._clustercenters

    @clustercenters.setter
    def clustercenters(self, val):
        self._clustercenters = np.asarray(val, dtype='float32', order='C')[:] if val is not None else None
        self._precentered = False

    @property
    def overwrite_dtrajs(self):
        """
        Should existing dtraj files be overwritten. Set this property to True to overwrite.
        """
        return self._overwrite_dtrajs

    @overwrite_dtrajs.setter
    def overwrite_dtrajs(self, value):
        self._overwrite_dtrajs = value

    @property
    #@alias('labels_') # TODO: for fully sklearn-compat this would have to be a flat array!
    def dtrajs(self):
        """Discrete trajectories (assigned data to cluster centers)."""
        if len(self._dtrajs) == 0:  # nothing assigned yet, doing that now
            self._dtrajs = self.assign(stride=1)

        return self._dtrajs  # returning what we have saved

    @property
    def index_clusters(self):
        """Returns trajectory/time indexes for all the clusters

        Returns
        -------
        indexes : list of ndarray( (N_i, 2) )
            For each state, all trajectory and time indexes where this cluster occurs.
            Each matrix has a number of rows equal to the number of occurrences of the corresponding state,
            with rows consisting of a tuple (i, t), where i is the index of the trajectory and t is the time index
            within the trajectory.
        """
        if len(self._dtrajs) == 0:  # nothing assigned yet, doing that now
            self._dtrajs = self.assign()

        if len(self._index_states) == 0: # has never been run
            self._index_states = index_states(self._dtrajs)

        return self._index_states

    def sample_indexes_by_cluster(self, clusters, nsample, replace=True):
        """Samples trajectory/time indexes according to the given sequence of states.

        Parameters
        ----------
        clusters : iterable of integers
            It contains the cluster indexes to be sampled

        nsample : int
            Number of samples per cluster. If replace = False, the number of returned samples per cluster could be smaller
            if less than nsample indexes are available for a cluster.

        replace : boolean, optional
            Whether the sample is with or without replacement

        Returns
        -------
        indexes : list of ndarray( (N, 2) )
            List of the sampled indices by cluster.
            Each element is an index array with a number of rows equal to N=len(sequence), with rows consisting of a
            tuple (i, t), where i is the index of the trajectory and t is the time index within the trajectory.
        """

        # Check if the catalogue (index_states)
        if len(self._index_states) == 0: # has never been run
            self._index_states = index_states(self.dtrajs)

        return sample_indexes_by_state(self._index_states[clusters], nsample, replace=replace)

    def _transform_array(self, X):
        """get closest index of point in :attr:`clustercenters` to x."""
        X = np.require(X, dtype=np.float32, requirements='C')
        # for performance reasons we pre-center the cluster centers for minRMSD.
        if self.metric == 'minRMSD' and not self._precentered:
            self._precentered = True

        model = ClusterModel(cluster_centers=self.clustercenters, metric=self.metric)
        dtraj = model.transform(X)
        res = dtraj[:, None]  # always return a column vector in this function
        return res

    def dimension(self):
        """output dimension of clustering algorithm (always 1)."""
        return 1

    def output_type(self):
        return np.int32()

    def assign(self, X=None, stride=1):
        """
        Assigns the given trajectory or list of trajectories to cluster centers by using the discretization defined
        by this clustering method (usually a Voronoi tesselation).

        You can assign multiple times with different strides. The last result of assign will be saved and is available
        as the attribute :func:`dtrajs`.

        Parameters
        ----------
        X : ndarray(T, n) or list of ndarray(T_i, n), optional, default = None
            Optional input data to map, where T is the number of time steps and n is the number of dimensions.
            When a list is provided they can have differently many time steps, but the number of dimensions need
            to be consistent. When X is not provided, the result of assign is identical to get_output(), i.e. the
            data used for clustering will be assigned. If X is given, the stride argument is not accepted.

        stride : int, optional, default = 1
            If set to 1, all frames of the input data will be assigned. Note that this could cause this calculation
            to be very slow for large data sets. Since molecular dynamics data is usually
            correlated at short timescales, it is often sufficient to obtain the discretization at a longer stride.
            Note that the stride option used to conduct the clustering is independent of the assign stride.
            This argument is only accepted if X is not given.

        Returns
        -------
        Y : ndarray(T, dtype=int) or list of ndarray(T_i, dtype=int)
            The discretized trajectory: int-array with the indexes of the assigned clusters, or list of such int-arrays.
            If called with a list of trajectories, Y will also be a corresponding list of discrete trajectories

        """
        if X is None:
            # if the stride did not change and the discrete trajectory is already present,
            # just return it
            if self._previous_stride is stride and len(self._dtrajs) > 0:
                return self._dtrajs
            self._previous_stride = stride
            skip = self.skip if hasattr(self, 'skip') else 0
            # map to column vectors
            mapped = self.get_output(stride=stride, chunk=self.chunksize, skip=skip)
            # flatten and save
            self._dtrajs = [np.transpose(m)[0] for m in mapped]
            # return
            return self._dtrajs
        else:
            if stride != 1:
                raise ValueError('assign accepts either X or stride parameters, but not both. If you want to map '+
                                 'only a subset of your data, extract the subset yourself and pass it as X.')
            # map to column vector(s)
            mapped = self.transform(X)
            # flatten
            if isinstance(mapped, np.ndarray):
                mapped = np.transpose(mapped)[0]
            else:
                mapped = [np.transpose(m)[0] for m in mapped]
            # return
            return mapped

    def save_dtrajs(self, trajfiles=None, prefix='',
                    output_dir='.',
                    output_format='ascii',
                    extension='.dtraj'):
        """saves calculated discrete trajectories. Filenames are taken from
        given reader. If data comes from memory dtrajs are written to a default
        filename.


        Parameters
        ----------
        trajfiles : list of str (optional)
            names of input trajectory files, will be used generate output files.
        prefix : str
            prepend prefix to filenames.
        output_dir : str
            save files to this directory.
        output_format : str
            if format is 'ascii' dtrajs will be written as csv files, otherwise
            they will be written as NumPy .npy files.
        extension : str
            file extension to append (eg. '.itraj')
        """
        if extension[0] != '.':
            extension = '.' + extension

        # obtain filenames from input (if possible, reader is a featurereader)
        if output_format == 'ascii':
            from msmtools.dtraj import write_discrete_trajectory as write_dtraj
        else:
            from msmtools.dtraj import save_discrete_trajectory as write_dtraj
        import os.path as path

        output_files = []

        if trajfiles is not None:  # have filenames available?
            for f in trajfiles:
                p, n = path.split(f)  # path and file
                basename, _ = path.splitext(n)
                if prefix != '':
                    name = "%s_%s%s" % (prefix, basename, extension)
                else:
                    name = "%s%s" % (basename, extension)
                # name = path.join(p, name)
                output_files.append(name)
        else:
            for i in range(len(self.dtrajs)):
                if prefix != '':
                    name = "%s_%i%s" % (prefix, i, extension)
                else:
                    name = str(i) + extension
                output_files.append(name)

        assert len(self.dtrajs) == len(output_files)

        if not os.path.exists(output_dir):
            mkdir_p(output_dir)

        for filename, dtraj in zip(output_files, self.dtrajs):
            dest = path.join(output_dir, filename)
            self.logger.debug('writing dtraj to "%s"' % dest)
            try:
                if path.exists(dest) and not self.overwrite_dtrajs:
                    raise EnvironmentError('Attempted to write dtraj "%s" which already existed. To automatically'
                                           ' overwrite existing files, set source.overwrite_dtrajs=True.' % dest)
                write_dtraj(dest, dtraj)
            except IOError:
                self.logger.exception('Exception during writing dtraj to "%s"' % dest)
