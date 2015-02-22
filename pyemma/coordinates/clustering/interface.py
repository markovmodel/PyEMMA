'''
Created on 18.02.2015

@author: marscher
'''
from pyemma.coordinates.transform.transformer import Transformer
from pyemma.util.log import getLogger
import numpy as np

log = getLogger('Clustering')


class AbstractClustering(Transformer):

    """
    provides a common interface for cluster algorithms.
    """

    def __init__(self):
        super(AbstractClustering, self).__init__()
        self.clustercenters = None
        self.dtrajs = []

    def map(self, x):
        """get closest index of point in :attr:`clustercenters` to x."""
        d = self.data_producer.distances(x, self.clustercenters)
        return np.argmin(d)

    def save_dtrajs(self, trajfiles=None, prefix='', output_format='ascii',
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
                name = path.join(p, prefix + '_' + basename + extension)
                output_files.append(name)
        else:
            for i in xrange(len(dtrajs)):
                if prefix is not '':
                    name = prefix + '_' + i + extension
                else:
                    name = i + extension
                output_files.append(name)

        assert len(dtrajs) == len(output_files)

        for filename, dtraj in zip(output_files, dtrajs):
            try:
                if path.exists(output_files):
                    # TODO: decide what to do if file already exists.
                    pass
                write_dtraj(filename, dtraj)
            except IOError:
                self.logger.exception(
                    "Exception during writing dtraj to '%s'" % filename)
