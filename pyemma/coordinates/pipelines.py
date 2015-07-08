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
from pyemma.coordinates.clustering.interface import AbstractClustering
from pyemma.coordinates.transform.transformer import Transformer
from pyemma.coordinates.data.feature_reader import FeatureReader

from pyemma.util.log import getLogger

__all__ = ['Discretizer',
           'Pipeline',
           ]

__author__ = 'noe, marscher'


class Pipeline(object):

    def __init__(self, chain, chunksize=100, param_stride=1):
        r"""Data processing pipeline.

        Parameters
        ----------
        chain : list of transformers like objects
            the order in the list defines the direction of data flow.
        chunksize : int, optional
            how many frames shall be processed at once.
        param_stride : int, optional
            omit every n'th data point

        """
        self._chain = []
        self.chunksize = chunksize
        self.param_stride = param_stride
        self.chunksize = chunksize

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
        r""" Appends a pipeline stage.

        Appends the given element to the end of the current chain.
        """
        if not isinstance(e, Transformer):
            raise TypeError("given element is not a transformer.")

        # set data producer
        if len(self._chain) == 0:
            data_producer = e
        else:
            data_producer = self._chain[-1]

        # avoid calling the setter of Transformer.data_producer, since this
        # triggers a re-parametrization even on readers (where it makes not sense)
        e._data_producer = data_producer
        e.chunksize = self.chunksize

        self._chain.append(e)

    def set_element(self, index, e):
        r""" Replaces a pipeline stage.

        Replace an element in chain and return replaced element.
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
        """deprecated. Identical to parametrize()
        """
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
        r"""
        Reads all data and discretizes it into discrete trajectories.
        """
        for element in self._chain:
            element.parametrize(stride=self.param_stride)

        self._parametrized = True

    def _is_parametrized(self):
        r"""
        Iterates through the pipeline elements and checks if every element is parametrized.
        """
        result = self._parametrized
        for el in self._chain:
            result &= el._parametrized
        return result


class Discretizer(Pipeline):

    r"""
    A Discretizer gets a FeatureReader, which extracts features (distances,
    angles etc.) of given trajectory data and passes this data in a memory
    efficient way through the given pipeline of a Transformer and Clustering.
    The clustering object is responsible for assigning the data to the cluster centers.

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
        r"""Saves calculated discrete trajectories. Filenames are taken from
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
