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

from abc import ABCMeta, abstractmethod
import numpy as np

from pyemma._base.loggable import Loggable
from pyemma.coordinates.data._base._in_memory_mixin import InMemoryMixin
from pyemma.util.types import is_int


class Iterable(InMemoryMixin, Loggable, metaclass=ABCMeta):
    _FALLBACK_CHUNKSIZE = 1000

    def __init__(self, chunksize=None):
        super(Iterable, self).__init__()
        self._default_chunksize = chunksize
        # should be set in subclass
        self._ndim = 0

    def dimension(self):
        return self._ndim

    @property
    def ndim(self):
        return self.dimension()

    @staticmethod
    def _compute_default_cs(dim, itemsize, logger=None):
        # obtain a human readable memory size from the config, convert it to bytes and calc maximum chunksize.
        from pyemma import config
        from pyemma.util.units import string_to_bytes
        max_bytes = string_to_bytes(config.default_chunksize)

        # TODO: consider rounding this to some cache size of CPU? e.g py-cpuinfo can obtain it.
        # if one time step is already bigger than max_memory, we set the chunksize to 1.
        bytes_per_frame = itemsize * dim
        max_frames = max(1, int(np.floor(max_bytes / bytes_per_frame)))
        assert max_frames * dim * itemsize <= max_bytes or max_frames == 1, \
            "number of frames times dim times sizeof(dtype) should be smaller or equal than max_bytes"
        result = max_frames

        assert result > 0
        if logger is not None:
            logger.debug('computed default chunksize to %s'
                         ' to limit memory per chunk to %s', result, config.default_chunksize)
        return result

    @property
    def default_chunksize(self):
        """ How much data will be processed at once, in case no chunksize has been provided.

        Notes
        -----
        This variable respects your setting for maximum memory in pyemma.config.default_chunksize
        """
        if self._default_chunksize is None:
            try:
                # TODO: if dimension is not yet fixed (eg tica var cutoff, use dim of data_producer.
                self.dimension()
                self.output_type()
            except:
                self._default_chunksize = Iterable._FALLBACK_CHUNKSIZE
            else:
                self._default_chunksize = Iterable._compute_default_cs(self.dimension(),
                                                                       self.output_type().itemsize, self.logger)
        return self._default_chunksize

    @property
    def chunksize(self):
        return self.default_chunksize

    @chunksize.setter
    def chunksize(self, value):
        if not isinstance(value, (type(None), int)):
            raise ValueError('chunksize has to be of type: None or int')
        if isinstance(value, int) and value < 0:
            raise ValueError("Chunksize of %s was provided, but has to be >= 0" % value)
        self._default_chunksize = value

    def iterator(self, stride=1, lag=0, chunk=None, return_trajindex=True, cols=None, skip=0):
        """ creates an iterator to stream over the (transformed) data.

        If your data is too large to fit into memory and you want to incrementally compute
        some quantities on it, you can create an iterator on a reader or transformer (eg. TICA)
        to avoid memory overflows.

        Parameters
        ----------

        stride : int, default=1
            Take only every stride'th frame.
        lag: int, default=0
            how many frame to omit for each file.
        chunk: int, default=None
            How many frames to process at once. If not given obtain the chunk size
            from the source.
        return_trajindex: boolean, default=True
            a chunk of data if return_trajindex is False, otherwise a tuple of (trajindex, data).
        cols: array like, default=None
            return only the given columns.
        skip: int, default=0
            skip 'n' first frames of each trajectory.

        Returns
        -------
        iter : instance of DataSourceIterator
            a implementation of a DataSourceIterator to stream over the data

        Examples
        --------

        >>> from pyemma.coordinates import source; import numpy as np
        >>> data = [np.arange(3), np.arange(4, 7)]
        >>> reader = source(data)
        >>> iterator = reader.iterator(chunk=1)
        >>> for array_index, chunk in iterator:
        ...     print(array_index, chunk)
        0 [[0]]
        0 [[1]]
        0 [[2]]
        1 [[4]]
        1 [[5]]
        1 [[6]]
        """
        if self.in_memory:
            from pyemma.coordinates.data.data_in_memory import DataInMemory
            return DataInMemory(self._Y).iterator(
                lag=lag, chunk=chunk, stride=stride, return_trajindex=return_trajindex, skip=skip
            )
        chunk = chunk if chunk is not None else self.chunksize
        if lag > 0:
            if chunk == 0 or lag <= chunk:
                it = self._create_iterator(skip=skip, chunk=chunk, stride=1,
                                           return_trajindex=return_trajindex, cols=cols)
                it.return_traj_index = True
                return _LaggedIterator(it, lag, return_trajindex, stride)
            else:
                it = self._create_iterator(skip=skip, chunk=chunk, stride=stride,
                                           return_trajindex=return_trajindex, cols=cols)
                it.return_traj_index = True
                it_lagged = self._create_iterator(skip=skip + lag, chunk=chunk, stride=stride,
                                                  return_trajindex=True, cols=cols)
                return _LegacyLaggedIterator(it, it_lagged, return_trajindex)
        return self._create_iterator(skip=skip, chunk=chunk, stride=stride,
                                     return_trajindex=return_trajindex, cols=cols)

    @abstractmethod
    def _create_iterator(self, skip=0, chunk=0, stride=1, return_trajindex=True, cols=None):
        """
        Should be implemented by non-abstract subclasses. Creates an instance-independent iterator.
        :param skip: How many frames to skip before streaming.
        :param chunk: The chunksize.
        :param stride: Take only every stride'th frame.
        :param return_trajindex: take the trajindex into account
        :return: a chunk of data if return_trajindex is False, otherwise a tuple of (trajindex, data).
        """
        raise NotImplementedError()

    def output_type(self):
        r""" By default transformers return single precision floats. """
        return np.float32()

    def __iter__(self):
        return self.iterator()


class _LaggedIterator(object):
    """ _LaggedIterator

    avoids double IO, by switching the chunksize on the given Iterable instance and
    remember an overlap.

    Parameters
    ----------
    it: instance of Iterable (stride=1)
    lag : int
        lag time
    actual_stride: int
        stride
    return_trajindex: bool
        whether to return the current trajectory index during iteration (itraj).
    """
    def __init__(self, it, lag, return_trajindex, actual_stride):
        self._it = it
        self._lag = lag
        assert is_int(lag)
        self._return_trajindex = return_trajindex
        self._overlap = None
        self._actual_stride = actual_stride
        self._sufficently_long_trajectories = [i for i, x in
                                               enumerate(self._it._data_source.trajectory_lengths(1, 0))
                                               if x > lag]
        self._max_size = max(x for x in self._it._data_source.trajectory_lengths(1, 0))

    @property
    def n_chunks(self):
        cs = self._it.chunksize
        n1 = self._it._data_source.n_chunks(cs, stride=self._actual_stride, skip=self._lag)
        n2 = self._it._data_source.n_chunks(cs, stride=self._actual_stride, skip=0)
        return min(n1, n2)

    def __len__(self):
        n1 = self._it._data_source.trajectory_lengths(self._actual_stride, 0).min()
        n2 = self._it._data_source.trajectory_lengths(self._actual_stride, self._lag).min()
        return min(n1, n2)

    def __iter__(self):
        return self

    @property
    def chunksize(self):
        return self._it.chunksize

    def reset(self):
        self._it.reset()

    def __next__(self):
        chunksize_old = self._it.chunksize
        try:
            self._it.chunksize = self._max_size if self.chunksize == 0 else self.chunksize
            changed = _skip_too_short_trajs(self._it, self._sufficently_long_trajectories)
            if changed:
                self._overlap = None

            if self._overlap is None:
                chunksize_old2 = self._it.chunksize
                try:
                    self._it.chunksize = self._lag
                    _, self._overlap = next(self._it)
                    assert len(self._overlap) <= self._lag, 'len(overlap) > lag... %s>%s' % (len(self._overlap), self._lag)
                    self._overlap = self._overlap[::self._actual_stride]
                finally:
                    self._it.chunksize = chunksize_old2

            chunksize_old3 = self._it.chunksize
            try:
                self._it.chunksize = self._it.chunksize * self._actual_stride
                itraj, data_lagged = next(self._it)
                assert len(data_lagged) <= self._it.chunksize * self._actual_stride
                frag = data_lagged[:min(self._it.chunksize - self._lag, len(data_lagged)), :]
                data = np.concatenate((self._overlap, frag[(self._actual_stride - self._lag)
                                                           % self._actual_stride::self._actual_stride]), axis=0)

                offset = min(self._it.chunksize - self._lag, len(data_lagged))
                self._overlap = data_lagged[offset::self._actual_stride, :]

                data_lagged = data_lagged[::self._actual_stride]
            finally:
                self._it.chunksize = chunksize_old3

            if self._it.last_chunk_in_traj:
                self._overlap = None

            if len(data) > len(data_lagged):
                # data chunk is bigger, truncate it to match data_lagged's shape
                data = data[:len(data_lagged)]
            elif len(data) < len(data_lagged):
                raise RuntimeError("chunk was smaller than time-lagged chunk (%s < %s), that should not happen!"
                                   % (len(data), len(data_lagged)))

            if self._return_trajindex:
                return itraj, data, data_lagged
            return data, data_lagged
        finally:
            self._it.chunksize = chunksize_old

    next = __next__

    def __enter__(self):
        self._it.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._it.__exit__(exc_type, exc_val, exc_tb)


class _LegacyLaggedIterator(object):
    """ _LegacyLaggedIterator uses two iterators to build time-lagged chunks.

    Parameters
    ----------
    it: DataSource, skip=0
    it_lagged: DataSource, skip=lag
    return_trajindex: bool
        whether to return the current trajectory index during iteration (itraj).
    """
    def __init__(self, it, it_lagged, return_trajindex):
        self._it = it
        self._it_lagged = it_lagged
        self._return_trajindex = return_trajindex
        self._sufficently_long_trajectories = [i for i, x in
                                               enumerate(self._it_lagged._data_source.trajectory_lengths(1, 0))
                                               if x > it_lagged.skip]

    @property
    def n_chunks(self):
        n1 = self._it.n_chunks
        n2 = self._it_lagged.n_chunks
        return min(n1, n2)

    @property
    def chunksize(self):
        return min(self._it.chunksize, self._it_lagged.chunksize)

    def __len__(self):
        return min(self._it.trajectory_lengths().min(), self._it_lagged.trajectory_lengths().min())

    def __iter__(self):
        return self

    def __next__(self):
        _skip_too_short_trajs(self._it_lagged, self._sufficently_long_trajectories)
        self._it._itraj = self._it_lagged._itraj

        itraj, data = self._it.next()
        assert itraj in self._sufficently_long_trajectories, itraj
        itraj_lag, data_lagged = self._it_lagged.next()
        assert itraj_lag in self._sufficently_long_trajectories, itraj_lag
        if itraj < itraj_lag:
            self._it._select_file(itraj_lag)
            itraj, data = self._it.next()
            assert not itraj > itraj_lag
        assert itraj == itraj_lag

        if len(data) > len(data_lagged):
            # data chunk is bigger, truncate it to match data_lagged's shape
            data = data[:len(data_lagged)]
        elif len(data) < len(data_lagged):
            raise RuntimeError("chunk was smaller than time-lagged chunk (%s < %s), that should not happen!"
                               % (len(data), len(data_lagged)))
        if self._return_trajindex:
            return itraj, data, data_lagged
        return data, data_lagged

    next = __next__

    def __enter__(self):
        self._it.__enter__()
        self._it_lagged.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._it.__exit__(exc_type, exc_val, exc_tb)
        self._it_lagged.__exit__(exc_type, exc_val, exc_tb)


def _skip_too_short_trajs(it, sufficiently_long_traj_indices):
    changed = False

    while (it._itraj not in sufficiently_long_traj_indices
           and it._itraj < it.state.ntraj):
        changed = True
        if len(sufficiently_long_traj_indices) == 1:
            if it._itraj > sufficiently_long_traj_indices[0]:
                raise StopIteration('no traj long enough.')
            it._select_file(sufficiently_long_traj_indices[0])
            break

        idx = (np.abs(np.array(sufficiently_long_traj_indices) - it._itraj)).argmin()
        if idx + 1 < len(sufficiently_long_traj_indices):
            next_itraj = sufficiently_long_traj_indices[idx + 1]
            it._select_file(next_itraj)
            assert it._itraj == next_itraj
        else:
            raise StopIteration('no trajectory long enough.')
    return changed

