__author__ = 'noe, marscher'

import numpy as np

from pyemma.coordinates.clustering.interface import AbstractClustering
from pyemma.coordinates.transform.transformer import Transformer
from pyemma.coordinates.data.feature_reader import FeatureReader

from pyemma.util.log import getLogger

__all__ = ['Discretizer',
           'Pipeline',
           ]


class Pipeline(object):

    def __init__(self, chain, chunksize=100, param_stride=1):
        """

        TODO:chunksize should be estimated from memory requirements (max memory usage)
        """
        self._chain = []
        self.chunksize = chunksize
        self.param_stride = param_stride

        # add given elements in chain
        for e in chain:
            self.add_element(e)

        self._parametrized = False

        name = "%s[%s]" % (self.__class__.__name__, hex(id(self)))
        self._logger = getLogger(name)

    @property
    def chunksize(self):
        return self._chunksize

    @chunksize.setter
    def chunksize(self, cs):
        self._chunksize = cs
        # update transformers to use new chunksize
        for e in self._chain:
            e.chunksize = cs

    def add_element(self, e):
        """
        appends the given element to the end of the current chain
        """
        # TODO: sanity checks on e

        # set data producer
        if len(self._chain) == 0:
            data_producer = e
        else:
            data_producer = self._chain[-1]

        e.data_producer = data_producer
        e.chunksize = self.chunksize

        self._chain.append(e)

    def set_element(self, index, e):
        """
        replace an element in chain and return replaced one.
        """
        if index > len(self._chain):
            raise IndexError("tried to access element %i, but chain has only %i"
                             " elements" % (index, len(self._chain)))

        if type(index) is not int:
            raise ValueError(
                "index is not a integer but '%s'" % str(type(index)))
        # if e is already in chain, we're finished
        if self._chain[index] is e:
            return

        # remove current index and its data producer
        replaced = self._chain.pop(index)
        replaced.data_producer = None

        self._chain.insert(index, e)

        if index == 0:
            e.data_producer = e
        else:
            # rewire data_producers
            e.data_producer = self._chain[index - 1]

        # if e has a successive element, need to set data_producer
        try:
            successor = self._chain[index + 1]
            successor.data_producer = e
        except IndexError:
            pass

        # set data_producer for predecessor of e
        # self._chain[max(0, index - 1)].data_producer = self._chain[index]

        # since data producer of element after insertion changed, reset its status
        # TODO: make parameterized a property?
        self._chain[index]._parameterized = False

        return replaced

    def run(self):
        import warnings
        warnings.warn(
            "run() is deprecated and will be disabled in the future. Use parametrize().", DeprecationWarning)
        self.parametrize()

    # TODO: DISCUSS - renamed run() to parametrize (because run is a bit ambiguous).
    # TODO: We could also call it fit() (here and in the transformers).
    # TODO: This might be nicer because it's shorter and the spelling is unambiguous
    # TODO: (in contrast to parametrize and parameterize and parameterise that
    # are all correct in english.
    def parametrize(self):
        """
        reads all data and discretizes it into discrete trajectories
        """
        for element in self._chain:
            element.parametrize(stride=self.param_stride)

        self._parametrized = True

    def _is_parametrized(self):
        """
        Iterates through the pipeline elements and check if every element is parametrized.
        """
        result = self._parametrized
        for el in self._chain:
            result &= el._parametrized
        return result

    def _estimate_chunksize_from_mem_requirement(self, reader):
        """
        estimate memory requirement from _chain of transformers and sets a
        chunksize accordingly
        """
        if not hasattr(reader, '_get_memory_per_frame'):
            self.chunksize = 0
            return

        try:
            import psutil
        except ImportError:
            self._logger.warning(
                "psutil not available. Can not estimate mem requirements")
            return

        M = psutil.virtual_memory()[1]  # available RAM in bytes
        self._logger.info("available RAM: %i" % M)
        const_mem = long(0)
        mem_per_frame = long(0)

        for trans in self._chain:
            mem_per_frame += trans._get_memory_per_frame()
            const_mem += trans._get_constant_memory()
        self._logger.info("per-frame memory requirements: %i" % mem_per_frame)

        # maximum allowed chunk size
        self._logger.info("const mem: %i" % const_mem)
        chunksize = (M - const_mem) / mem_per_frame
        if chunksize < 0:
            raise MemoryError(
                'Not enough memory for desired transformation _chain!')

        # is this chunksize sufficient to store full trajectories?
        chunksize = min(chunksize, np.max(reader.trajectory_lengths()))
        self._logger.info("resulting chunk size: %i" % chunksize)

        # set chunksize
        self.chunksize = chunksize

        # any memory unused? if yes, we can store results
        Mfree = M - const_mem - chunksize * mem_per_frame
        self._logger.info("free memory: %i" % Mfree)

        # starting from the back of the pipeline, store outputs if possible
        for trans in reversed(self._chain):
            mem_req_trans = trans.n_frames_total() * \
                trans._get_memory_per_frame()
            if Mfree > mem_req_trans:
                Mfree -= mem_req_trans
                # TODO: before we are allowed to call this method, we have to ensure all memory requirements are correct
                # trans.operate_in_memory()
                self._logger.info("spending %i bytes to operate in main memory: %s "
                                  % (mem_req_trans,  trans.describe()))


class Discretizer(Pipeline):

    """
    A Discretizer gets a FeatureReader, which extracts features (distances,
    angles etc.) of given trajectory data and passes this data in a memory
    efficient way through the given pipeline of a Transformer and a clustering.
    The clustering object is responsible for assigning the data to discrete
    states.

    Parameters
    ----------
    reader : a FeatureReader object
        reads trajectory data and selects features.
    transform : a Transformer object (optional)
        the Transformer will be used to e.g reduce dimensionality of inputs.
    cluster : a clustering object
        used to assign input data to discrete states/ discrete trajectories.
    chunksize : int, optional
        how many frames shall be processed at once.
    """

    def __init__(self, reader, transform=None, cluster=None, chunksize=100, param_stride=1):
        # init with an empty chain and add given transformers afterwards
        Pipeline.__init__(
            self, [], chunksize=chunksize, param_stride=param_stride)

        # check input
        if not isinstance(reader, Transformer):
            raise ValueError('given reader is not of the correct type')
        else:
            if reader.data_producer is not reader:
                raise ValueError("given reader is not a first stance data source."
                                 " Check if its a FeatureReader or DataInMemory")
        if transform is not None:
            if not isinstance(transform, Transformer):
                raise ValueError('transform is not a transformer but "%s"' %
                                 str(type(transform)))
        if cluster is None:
            raise ValueError('Must specify a clustering algorithm!')
        else:
            assert isinstance(cluster, AbstractClustering), \
                'cluster is not of the correct type'

        if hasattr(reader, 'featurizer'):  # reader is a FeatureReader
            if reader.featurizer.dimension == 0:
                self._logger.warning("no features selected!")

        self.add_element(reader)

        if transform is not None:
            self.add_element(transform)

        self.add_element(cluster)

# currently heuristical chunksize estimation is turned off.
#         if chunksize is not None:
#             build_chain(self.transformers, chunksize)
#             self._chunksize = chunksize
#         else:
#             self._chunksize = None
#             build_chain(self.transformers)
#             self._estimate_chunksize_from_mem_requirement(reader)

        self._parametrized = False

    @property
    def dtrajs(self):
        """ get discrete trajectories """
        if not self._parametrized:
            self._logger.info("not yet parametrized, running now.")
            self.parametrize()
        return self._chain[-1].dtrajs

    def save_dtrajs(self, prefix='', output_dir='.',
                    output_format='ascii', extension='.dtraj'):
        """saves calculated discrete trajectories. Filenames are taken from
        given reader. If data comes from memory dtrajs are written to a default
        filename.


        Parameters
        ----------
        prefix : str
            prepend prefix to filenames.
        output_dir : str (optional)
            save files to this directory. Defaults to current working directory.
        output_format : str
            if format is 'ascii' dtrajs will be written as csv files, otherwise
            they will be written as NumPy .npy files.
        extension : str
            file extension to append (eg. '.itraj')

        """

        clustering = self._chain[-1]
        reader = self._chain[0]

        assert isinstance(clustering, AbstractClustering)

        trajfiles = None
        if isinstance(reader, FeatureReader):
            trajfiles = reader.trajfiles

        clustering.save_dtrajs(
            trajfiles, prefix, output_dir, output_format, extension)
