__author__ = 'noe'

import psutil
import numpy as np

from pyemma.coordinates.transform.transformer import Transformer
from pyemma.coordinates.io.reader import ChunkedReader
from pyemma.coordinates.util.chaining import build_chain

from pyemma.util.log import getLogger


logger = getLogger('Discretizer')
__all__ = ['Discretizer']


class Discretizer(object):

    """
    A Discretizer gets a FeatureReader, which defines the features (distances,
    angles etc.) of given trajectory data and passes this data in a memory
    efficient way through the given pipeline of a Transformer and a clustering.
    The clustering object is responsible for assigning the data to discrete
    states.

    Currently the constructor will calculate everything instantly.


    Parameters
    ----------
    reader : a FeatureReader object
        reads trajectory data and selects features.
    transform : a Transformer object (optional)
        the Transformer will be used to e.g reduce dimensionality of inputs.
    cluster : a clustering object
        used to assign input data to discrete states/ discrete trajectories.
    """

    def __init__(self, reader, transform=None, cluster=None):
        # check input
        assert isinstance(reader, ChunkedReader), \
            'reader is not of the correct type'
        if (transform is not None):
            assert isinstance(transform, Transformer), \
                'transform is not of the correct type'
        if cluster is None:
            raise ValueError('Must specify a clustering algorithm!')
        else:
            assert isinstance(cluster, Transformer), \
                'cluster is not of the correct type'

        if reader.featurizer.dimension == 0:
            logger.warning("no features selected!")

        self.transformers = [reader]

        if transform is not None:
            self.transformers.append(transform)

        self.transformers.append(cluster)

        self._parameterized = False
        self._chunksize = None

        build_chain(self.transformers)

        self._estimate_chunksize_from_mem_requirement(reader)

    def run(self):
        """
        reads all data and discretizes it into discrete trajectories
        """
        for trans in self.transformers:
            trans.parametrize()

        self._parameterized = True

    @property
    def dtrajs(self):
        """ get discrete trajectories """
        if not self._parameterized:
            logger.info("not yet parametrized, running now.")
            self.run()
        return self.transformers[-1].dtrajs

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, cs):
        self._chunksize = cs
        # update transformers to use new chunksize
        for trans in self.transformers:
            trans.chunksize = cs

    def _estimate_chunksize_from_mem_requirement(self, reader):
        """
        estimate memory requirement from chain of transformers and sets a
        chunksize accordingly
        """

        M = psutil.virtual_memory()[1]  # available RAM in bytes
        logger.info("available RAM: %i" % M)
        const_mem = long(0)
        mem_per_frame = long(0)

        for trans in self.transformers:
            mem_per_frame += trans.get_memory_per_frame()
            const_mem += trans.get_constant_memory()
        logger.info("per-frame memory requirements: %i" % mem_per_frame)

        # maximum allowed chunk size
        logger.info("const mem: %i" % const_mem)
        chunksize = (M - const_mem) / mem_per_frame
        if chunksize < 0:
            raise MemoryError(
                'Not enough memory for desired transformation chain!')

        # is this chunksize sufficient to store full trajectories?
        chunksize = min(chunksize, np.max(reader.trajectory_lengths()))
        logger.info("resulting chunk size: %i" % chunksize)

        # set chunksize
        self.chunksize = chunksize

        # any memory unused? if yes, we can store results
        Mfree = M - const_mem - chunksize * mem_per_frame
        logger.info("free memory: %i" % Mfree)

        # starting from the back of the pipeline, store outputs if possible
        for trans in reversed(self.transformers):
            mem_req_trans = trans.n_frames_total() * \
                trans.get_memory_per_frame()
            if Mfree > mem_req_trans:
                Mfree -= mem_req_trans
                # TODO: before we are allowed to call this method, we have to ensure all memory requirements are correct!
                # trans.operate_in_memory()
                logger.info("spending %i bytes to operate in main memory: %s "
                            % (mem_req_trans,  trans.describe()))
