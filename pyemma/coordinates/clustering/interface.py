# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
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
'''
Created on 18.02.2015

@author: marscher
'''
from pyemma.coordinates.transform.transformer import Transformer
from pyemma.util.files import mkdir_p

import numpy as np
import os

from pyemma.coordinates.clustering import regspatial


class AbstractClustering(Transformer):

    """
    provides a common interface for cluster algorithms.
    """

    def __init__(self, metric='euclidean'):
        super(AbstractClustering, self).__init__()
        self.metric = metric
        self.clustercenters = None
        self._previous_stride = -1
        self._dtrajs = []
        self._overwrite_dtrajs = False

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
    def dtrajs(self):
        """Discrete trajectories (assigned data to cluster centers)."""
        if len(self._dtrajs) == 0:  # nothing assigned yet, doing that now
            self._dtrajs = self.assign(stride=1)

        return self._dtrajs  # returning what we have saved

    def _map_array(self, X):
        """get closest index of point in :attr:`clustercenters` to x."""
        dtraj = np.empty(X.shape[0], dtype=self.output_type())
        regspatial.assign(X.astype(np.float32, order='C', copy=False),
                          self.clustercenters, dtraj, self.metric)
        res = dtraj[:, None]  # always return a column vector in this function
        return res

    def dimension(self):
        """output dimension of clustering algorithm (always 1)."""
        return 1

    #@doc_inherit
    # TODO: inheritance of docstring should work
    def output_type(self):
        return np.int32

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
            # map to column vectors
            mapped = self.get_output(stride=stride)
            # flatten and save
            self._dtrajs = [np.transpose(m)[0] for m in mapped]
            # return
            return self._dtrajs
        else:
            if stride != 1:
                raise ValueError('assign accepts either X or stride parameters, but not both. If you want to map '+
                                 'only a subset of your data, extract the subset yourself and pass it as X.')
            # map to column vector(s)
            mapped = self.map(X)
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
            from pyemma.msm.io import write_discrete_trajectory as write_dtraj
        else:
            from pyemma.msm.io import save_discrete_trajectory as write_dtraj
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
            for i in xrange(len(self.dtrajs)):
                if prefix is not '':
                    name = "%s_%i%s" % (prefix, i, extension)
                else:
                    name = str(i) + extension
                output_files.append(name)

        assert len(self.dtrajs) == len(output_files)

        if not os.path.exists(output_dir):
            mkdir_p(output_dir)

        for filename, dtraj in zip(output_files, self.dtrajs):
            dest = path.join(output_dir, filename)
            self._logger.debug('writing dtraj to "%s"' % dest)
            try:
                if path.exists(dest) and not self.overwrite_dtrajs:
                    raise EnvironmentError('Attempted to write dtraj "%s" which already existed. To automatically'
                                           'overwrite existing files, set source.overwrite_dtrajs=True.' % dest)
                write_dtraj(dest, dtraj)
            except IOError:
                self._logger.exception('Exception during writing dtraj to "%s"' % dest)
