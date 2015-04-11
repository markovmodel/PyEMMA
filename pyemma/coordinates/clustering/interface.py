'''
Created on 18.02.2015

@author: marscher
'''
from pyemma.coordinates.transform.transformer import Transformer
from pyemma.util.files import mkdir_p

import numpy as np
import os

from pyemma.coordinates.clustering import regspatial
from pyemma.util.annotators import doc_inherit


class AbstractClustering(Transformer):

    """
    provides a common interface for cluster algorithms.
    """

    def __init__(self, metric='euclidean'):
        super(AbstractClustering, self).__init__()
        self.metric = metric
        self.clustercenters = None
        self.dtrajs = []

    def _map_array(self, X):
        """get closest index of point in :attr:`clustercenters` to x."""
        dtraj = np.empty(X.shape[0], np.int64)
        regspatial.assign(X.astype(np.float32, order='C', copy=False),
                          self.clustercenters, dtraj, self.metric)
        return dtraj

    @doc_inherit
    def dimension(self):
        return 1

    def assign(self, X):
        """
        Assigns the given trajectory or list of trajectories to cluster centers by using the discretization defined
        by this clustering method (usually a Voronoi tesselation)

        Parameters
        ----------
        X : ndarray(T, n) or list of ndarray(T_i, n)
            The input data, where T is the number of time steps and n is the number of dimensions.
            When a list is provided they can have differently many time steps, but the number of dimensions need
            to be consistent.

        Returns
        -------
        Y : ndarray(T, dtype=int) or list of ndarray(T_i, dtype=int)
            The discretized trajectory: int-array with the indexes of the assigned clusters, or list of such int-arrays.
            If called with a list of trajectories, Y will also be a corresponding list of discrete trajectories

        """
        return self.map(X)

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

        dtrajs = self.dtrajs  # clustering.dtrajs
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
            for i in xrange(len(dtrajs)):
                if prefix is not '':
                    name = "%s_%i%s" % (prefix, i, extension)
                else:
                    name = i + extension
                output_files.append(name)

        assert len(dtrajs) == len(output_files)

        if not os.path.exists(output_dir):
            mkdir_p(output_dir)

        for filename, dtraj in zip(output_files, dtrajs):
            dest = path.join(output_dir, filename)
            self._logger.debug('writing dtraj to "%s"' % dest)
            try:
                if path.exists(dest):
                    # TODO: decide what to do if file already exists.
                    self._logger.warn('overwriting existing dtraj "%s"' % dest)
                    pass
                write_dtraj(dest, dtraj)
            except IOError:
                self._logger.exception('Exception during writing dtraj to "%s"' % dest)
