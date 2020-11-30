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
from math import ceil

import numpy as np

from pyemma.coordinates.data._base.iterable import Iterable
from pyemma.coordinates.data._base.random_accessible import TrajectoryRandomAccessible
from pyemma.util import config

import os


class DataSource(Iterable, TrajectoryRandomAccessible):
    """
    Superclass for all pipeline elements. It inherits "Iterable", therefore serves as an iterator factory over the
    data it holds. The difference to Iterable is that DataSource is specialized for trajectories, whereas the concept
    of trajectories is generally unknown for Iterable.
    """
    _serialize_version = 0
    __serialize_fields = ('_is_reader', ) # other private fields are not needed, because they are set by child impl ctors.

    def __init__(self, chunksize=None):
        super(DataSource, self).__init__(chunksize=chunksize)

        # following properties have to be set in subclass
        self._lengths = []
        self._offsets = []
        self._filenames = None
        self._is_reader = False

    @property
    def ntraj(self):
        if self._is_reader:
            assert hasattr(self, '_ntraj')
            return self._ntraj

        return self.data_producer.ntraj

    @property
    def filenames(self):
        """ list of file names the data is originally being read from.

        Returns
        -------
        names : list of str
            list of file names at the beginning of the input chain.
        """
        if self._is_reader:
            assert self._filenames is not None
            return self._filenames
        else:
            return self.data_producer.filenames

    @filenames.setter
    def filenames(self, filename_list):

        if isinstance(filename_list, str):
            filename_list = [filename_list]

        uniq = set(filename_list)
        if len(uniq) != len(filename_list):
            self.logger.warning("duplicate files/arrays detected")
            filename_list = list(uniq)

        from pyemma.coordinates.data.data_in_memory import DataInMemory

        if self._is_reader:
            if isinstance(self, DataInMemory):
                import warnings
                warnings.warn('filenames are not being used for DataInMemory')
                return

            self._ntraj = len(filename_list)
            if self._ntraj == 0:
                raise ValueError("empty file list")

            # validate files
            for f in filename_list:
                try:
                    stat = os.stat(f)
                except EnvironmentError:
                    self.logger.exception('Error during access of file "%s"' % f)
                    raise ValueError('could not read file "%s"' % f)

                if not os.path.isfile(f): # can be true for symlinks to directories
                    raise ValueError('"%s" is not a valid file')

                if stat.st_size == 0:
                    raise ValueError('file "%s" is empty' % f)

            # number of trajectories/data sets
            self._filenames = filename_list
            # determine len and dim via cache lookup,
            lengths = []
            offsets = []
            ndims = []
            # avoid cyclic imports
            from pyemma.coordinates.data.util.traj_info_cache import TrajectoryInfoCache
            from pyemma._base.progress import ProgressReporter
            pg = ProgressReporter()
            pg.register(len(filename_list), 'Obtaining file info')
            with pg.context():
                for filename in filename_list:
                    if config.use_trajectory_lengths_cache:
                        info = TrajectoryInfoCache.instance()[filename, self]
                    else:
                        info = self._get_traj_info(filename)
                    # nested data set support.
                    if hasattr(info, 'children'):
                        lengths.append(info.length)
                        offsets.append(info.offsets)
                        ndims.append(info.ndim)
                        for c in info.children:
                            lengths.append(c.length)
                            offsets.append(c.offsets)
                            ndims.append(c.ndim)
                    else:
                        lengths.append(info.length)
                        offsets.append(info.offsets)
                        ndims.append(info.ndim)
                    if len(filename_list) > 3:
                        pg.update(1)

            # ensure all trajs have same dim
            if not np.unique(ndims).size == 1:
                # group files by their dimensions to give user indicator
                ndims = np.array(ndims)
                filename_list = np.asarray(filename_list)
                sort_inds = np.argsort(ndims)
                import itertools, operator
                res = {}
                for dim, files in itertools.groupby(zip(ndims[sort_inds], filename_list[sort_inds]),
                                                    operator.itemgetter(0)):
                    res[dim] = list(f[1] for f in files)

                raise ValueError("Input data has different dimensions ({dims})!"
                                 " Files grouped by dimensions: {groups}".format(dims=res.keys(),
                                                                                 groups=res))

            self._ndim = ndims[0]
            self._lengths = lengths
            self._offsets = offsets

        else:
            # propagate this until we finally have a a reader
            self.data_producer.filenames = filename_list

    def _get_traj_info(self, filename):
        raise NotImplementedError

    @property
    def is_reader(self):
        """
        Property telling if this data source is a reader or not.
        Returns
        -------
        bool: True if this data source is a reader and False otherwise
        """
        return self._is_reader

    @property
    def data_producer(self):
        """
        The data producer for this data source object (can be another data source object).
        Returns
        -------
        This data source's data producer.
        """
        return self

    def _data_flow_chain(self):
        """
        Get a list of all elements in the data flow graph.
        The first element is the original source, the next one reads from the prior and so on and so forth.

        Returns
        -------
        list: list of data sources

        """
        if self.data_producer is None:
            return []

        res = []
        ds = self.data_producer
        while not ds.is_reader:
            res.append(ds)
            ds = ds.data_producer
        res.append(ds)
        res = res[::-1]
        return res

    @staticmethod
    def _chunk_finite(data):
        if isinstance(data, np.ndarray):
            return np.isfinite(data)
        elif hasattr(data, 'xyz'):
            return np.isfinite(data.xyz)
        return True

    def _source_from_memory(self, data_producer=None):
        from pyemma.coordinates.data import DataInMemory
        if data_producer is None:
            data_producer = self
        while data_producer is not data_producer.data_producer:
            if isinstance(data_producer, DataInMemory): return True
            data_producer = data_producer.data_producer
        return isinstance(data_producer, DataInMemory)

    def number_of_trajectories(self, stride=None):
        r""" Returns the number of trajectories.

        Parameters
        ----------
        stride: None (default) or np.ndarray

        Returns
        -------
            int : number of trajectories
        """
        if not IteratorState.is_uniform_stride(stride):
            n = len(np.unique(stride[:, 0]))
        else:
            n = self.ntraj
        return n

    def trajectory_length(self, itraj, stride=1, skip=0):
        r"""Returns the length of trajectory of the requested index.

        Parameters
        ----------
        itraj : int
            trajectory index
        stride : int
            return value is the number of frames in the trajectory when
            running through it with a step size of `stride`.
        skip: int or None
            skip n frames.

        Returns
        -------
        int : length of trajectory
        """
        if itraj >= self.ntraj:
            raise IndexError("given index (%s) exceeds number of data sets (%s)."
                             " Zero based indexing!" % (itraj, self.ntraj))
        if not IteratorState.is_uniform_stride(stride):
            selection = stride[stride[:, 0] == itraj][:, 0]
            return 0 if itraj not in selection else len(selection)
        else:
            res = max((self._lengths[itraj] - skip - 1) // int(stride) + 1, 0)
            return res

    def n_chunks(self, chunksize, stride=1, skip=0):
        """ how many chunks an iterator of this sourcde will output, starting (eg. after calling reset())

        Parameters
        ----------
        chunksize
        stride
        skip
        """
        if chunksize != 0:
            chunksize = float(chunksize)
            chunks = int(sum((ceil(l / chunksize) for l in self.trajectory_lengths(stride=stride, skip=skip))))
        else:
            chunks = self.number_of_trajectories(stride)
        return chunks

    def trajectory_lengths(self, stride=1, skip=0):
        r""" Returns the length of each trajectory.

        Parameters
        ----------
        stride : int
            return value is the number of frames of the trajectories when
            running through them with a step size of `stride`.
        skip : int
            skip parameter

        Returns
        -------
        array(dtype=int) : containing length of each trajectory
        """
        n = self.ntraj

        if not IteratorState.is_uniform_stride(stride):
            return np.fromiter((self.trajectory_length(itraj, stride)
                                for itraj in range(n)),
                               dtype=int, count=n)
        else:
            return np.fromiter((self.trajectory_length(itraj, stride, skip)
                                for itraj in range(n)),
                               dtype=int, count=n)

    def n_frames_total(self, stride=1, skip=0):
        r"""Returns total number of frames.

        Parameters
        ----------
        stride : int
            return value is the number of frames in trajectories when
            running through them with a step size of `stride`.
        skip : int, default=0
            skip the first initial n frames per trajectory.
        Returns
        -------
        n_frames_total : int
            total number of frames.
        """
        if not IteratorState.is_uniform_stride(stride):
            return len(stride)

        return sum(self.trajectory_lengths(stride=stride, skip=skip))

    # workers
    def get_output(self, dimensions=slice(0, None), stride=1, skip=0, chunk=None):
        """Maps all input data of this transformer and returns it as an array or list of arrays

        Parameters
        ----------
        dimensions : list-like of indexes or slice, default=all
           indices of dimensions you like to keep.
        stride : int, default=1
           only take every n'th frame.
        skip : int, default=0
            initially skip n frames of each file.
        chunk: int, default=None
            How many frames to process at once. If not given obtain the chunk size
            from the source.

        Returns
        -------
        output : list of ndarray(T_i, d)
           the mapped data, where T is the number of time steps of the input data, or if stride > 1,
           floor(T_in / stride). d is the output dimension of this transformer.
           If the input consists of a list of trajectories, Y will also be a corresponding list of trajectories

        """
        if isinstance(dimensions, int):
            ndim = 1
            dimensions = slice(dimensions, dimensions + 1)
        elif isinstance(dimensions, (list, np.ndarray, tuple, slice)):
            if hasattr(dimensions, 'ndim') and dimensions.ndim > 1:
                raise ValueError('dimension indices can\'t have more than one dimension')
            ndim = len(np.zeros(self.ndim)[dimensions])
        else:
            raise ValueError('unsupported type (%s) of "dimensions"' % type(dimensions))

        assert ndim > 0, "ndim was zero in %s" % self.__class__.__name__

        if chunk is None:
            chunk = self.chunksize

        # create iterator
        if self.in_memory and not self._mapping_to_mem_active:
            from pyemma.coordinates.data.data_in_memory import DataInMemory
            assert self._Y is not None
            it = DataInMemory(self._Y)._create_iterator(skip=skip, chunk=chunk,
                                                        stride=stride, return_trajindex=True)
        else:
            it = self._create_iterator(skip=skip, chunk=chunk, stride=stride, return_trajindex=True)

        with it:
            # allocate memory
            try:
                from pyemma import config
                if config.coordinates_check_output:
                    trajs = [np.full((l, ndim), np.nan, dtype=self.output_type()) for l in it.trajectory_lengths()]
                else:
                    # TODO: avoid having a copy here, if Y is already filled
                    trajs = [np.empty((l, ndim), dtype=self.output_type())
                             for l in it.trajectory_lengths()]
            except MemoryError:
                self.logger.exception("Could not allocate enough memory to map all data."
                                      " Consider using a larger stride.")
                return

            if self._logger_is_active(self._loglevel_DEBUG):
                self.logger.debug("get_output(): dimensions=%s" % str(dimensions))
                self.logger.debug("get_output(): created output trajs with shapes: %s"
                                   % [x.shape for x in trajs])
                self.logger.debug("nchunks :%s, chunksize=%s" % (it.n_chunks, it.chunksize))
            # fetch data
            from pyemma._base.progress import ProgressReporter
            pg = ProgressReporter()
            pg.register(it.n_chunks, description='getting output of %s' % self.__class__.__name__)
            with pg.context(), it:
                for itraj, chunk in it:
                    i = slice(it.pos, it.pos + len(chunk))
                    assert i.stop - i.start > 0
                    trajs[itraj][i, :] = chunk[:, dimensions]
                    pg.update(1)

        if config.coordinates_check_output:
            for i, t in enumerate(trajs):
                finite = self._chunk_finite(t)
                if not np.all(finite):
                    # determine position
                    frames = np.where(np.logical_not(finite))
                    if not len(frames):
                        raise RuntimeError('nothing got assigned for traj {}'.format(i))
                    raise RuntimeError('unassigned sections in traj {i} in range [{frames}]'.format(frames=frames, i=i))

        return trajs

    def write_to_hdf5(self, filename, group='/', data_set_prefix='', overwrite=False,
                      stride=1, chunksize=None, h5_opt=None):
        """ writes all data of this Iterable to a given HDF5 file.
        This is equivalent of writing the result of func:`pyemma.coordinates.data._base.DataSource.get_output` to a file.

        Parameters
        ----------
        filename: str
            file name of output HDF5 file
        group: str, default='/'
            write all trajectories to this HDF5 group. The group name may not already exist in the file.
        data_set_prefix: str, default=None
            data set name prefix, will postfixed with the index of the trajectory.
        overwrite: bool, default=False
            if group and data sets already exist, shall we overwrite data?
        stride: int, default=1
            stride argument to iterator
        chunksize: int, default=None
            how many frames to process at once
        h5_opt: dict
            optional parameters for h5py.create_dataset

        Notes
        -----
        You can pass the following via h5_opt to enable compression/filters/shuffling etc:

        chunks
            (Tuple) Chunk shape, or True to enable auto-chunking.
        maxshape
            (Tuple) Make the dataset resizable up to this shape.  Use None for
            axes you want to be unlimited.
        compression
            (String or int) Compression strategy.  Legal values are 'gzip',
            'szip', 'lzf'.  If an integer in range(10), this indicates gzip
            compression level. Otherwise, an integer indicates the number of a
            dynamically loaded compression filter.
        compression_opts
            Compression settings.  This is an integer for gzip, 2-tuple for
            szip, etc. If specifying a dynamically loaded compression filter
            number, this must be a tuple of values.
        scaleoffset
            (Integer) Enable scale/offset filter for (usually) lossy
            compression of integer or floating-point data. For integer
            data, the value of scaleoffset is the number of bits to
            retain (pass 0 to let HDF5 determine the minimum number of
            bits necessary for lossless compression). For floating point
            data, scaleoffset is the number of digits after the decimal
            place to retain; stored values thus have absolute error
            less than 0.5*10**(-scaleoffset).
        shuffle
            (T/F) Enable shuffle filter. Only effective in combination with chunks.
        fletcher32
            (T/F) Enable fletcher32 error detection. Not permitted in
            conjunction with the scale/offset filter.
        fillvalue
            (Scalar) Use this value for uninitialized parts of the dataset.
        track_times
            (T/F) Enable dataset creation timestamps.
        """
        if h5_opt is None:
            h5_opt = {}
        import h5py
        from pyemma._base.progress import ProgressReporter
        pg = ProgressReporter()
        it = self.iterator(stride=stride, chunk=chunksize, return_trajindex=True)
        pg.register(it.n_chunks, 'writing output')
        with h5py.File(filename, mode='a') as f, it, pg.context():
            if group not in f:
                g = f.create_group(group)
            elif group == '/':  # root always exists.
                g = f[group]
            elif group in f and overwrite:
                self.logger.info('overwriting group "{}"'.format(group))
                del f[group]
                g = f.create_group(group)
            else:
                raise ValueError('Given group "{}" already exists. Choose another one.'.format(group))

            # check output data sets
            data_sets = {}
            for itraj in np.arange(self.ntraj):
                template = '{prefix}_{index}' if data_set_prefix else '{index}'
                ds_name = template.format(prefix=data_set_prefix, index='{:04d}'.format(itraj))
                # group can be reused, eg. was empty before now check if we will overwrite something
                if ds_name in g:
                    if not overwrite:
                        raise ValueError('Refusing to overwrite data in group "{}".'.format(group))
                else:
                    data_sets[itraj] = g.require_dataset(ds_name, shape=(self.trajectory_length(itraj=itraj, stride=stride),
                                                                         self.ndim), dtype=self.output_type(), **h5_opt)
            for itraj, X in it:
                ds = data_sets[itraj]
                ds[it.pos:it.pos + len(X)] = X
                pg.update(1)

    def write_to_csv(self, filename=None, extension='.dat', overwrite=False,
                     stride=1, chunksize=None, **kw):
        """ write all data to csv with numpy.savetxt

        Parameters
        ----------
        filename : str, optional
            filename string, which may contain placeholders {itraj} and {stride}:

            * itraj will be replaced by trajetory index
            * stride is stride argument of this method

            If filename is not given, it is being tried to obtain the filenames
            from the data source of this iterator.
        extension : str, optional, default='.dat'
            filename extension of created files
        overwrite : bool, optional, default=False
            shall existing files be overwritten? If a file exists, this method will raise.
        stride : int
            omit every n'th frame
        chunksize: int, default=None
            how many frames to process at once
        kw : dict, optional
            named arguments passed into numpy.savetxt (header, seperator etc.)

        Example
        -------
        Assume you want to save features calculated by some FeatureReader to ASCII:

        >>> import numpy as np, pyemma
        >>> import os
        >>> from pyemma.util.files import TemporaryDirectory
        >>> from pyemma.util.contexts import settings
        >>> data = [np.random.random((10,3))] * 3
        >>> reader = pyemma.coordinates.source(data)
        >>> filename = "distances_{itraj}.dat"
        >>> with TemporaryDirectory() as td, settings(show_progress_bars=False):
        ...    out = os.path.join(td, filename)
        ...    reader.write_to_csv(out, header='', delimiter=';')
        ...    print(sorted(os.listdir(td)))
        ['distances_0.dat', 'distances_1.dat', 'distances_2.dat']
        """
        import os
        if not filename:
            assert hasattr(self, 'filenames')
            #    raise RuntimeError("could not determine filenames")
            filenames = []
            for f in self.filenames:
                base, _ = os.path.splitext(f)
                filenames.append(base + extension)
        elif isinstance(filename, str):
            filename = filename.replace('{stride}', str(stride))
            filenames = [filename.replace('{itraj}', str(itraj)) for itraj
                         in range(self.number_of_trajectories())]
        else:
            raise TypeError("filename should be str or None")
        self.logger.debug("write_to_csv, filenames=%s" % filenames)
        # check files before starting to write
        import errno
        for f in filenames:
            try:
                st = os.stat(f)
                raise OSError(errno.EEXIST)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    if overwrite:
                        continue
                elif e.errno == errno.ENOENT:
                    continue
                raise
        f = None
        from pyemma._base.progress import ProgressReporter
        pg = ProgressReporter()
        it = self.iterator(stride, chunk=chunksize, return_trajindex=False)
        pg.register(it.n_chunks, "saving to csv")
        with it, pg.context():
            oldtraj = -1
            for X in it:
                if oldtraj != it.current_trajindex:
                    if f is not None:
                        f.close()
                    fn = filenames[it.current_trajindex]
                    self.logger.debug("opening file %s for writing csv." % fn)
                    f = open(fn, 'wb')
                    oldtraj = it.current_trajindex
                np.savetxt(f, X, **kw)
                f.flush()
                pg.update(1, 0)
        if f is not None:
            f.close()


class IteratorState(object):
    """
    State class holding all the relevant information of an iterator's state.
    """

    def __init__(self, skip=0, chunk=0, return_trajindex=False, ntraj=0, cols=None):
        self.skip = skip
        self.chunk = chunk
        self.return_trajindex = return_trajindex
        self.itraj = 0
        self.ntraj = ntraj
        self.t = 0
        self._pos = 0
        self.pos_adv = 0
        self.stride = None
        self.uniform_stride = False
        self.traj_keys = None
        self.trajectory_lengths = None
        self.ra_indices_for_traj_dict = {}
        self.cols = cols
        self.current_itraj = 0

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, value):
        self._pos = value

    def ra_indices_for_traj(self, traj):
        """
        Gives the indices for a trajectory file index (without changing the order within the trajectory itself).
        :param traj: a trajectory file index
        :return: a Nx1 - np.array of the indices corresponding to the trajectory index
        """
        assert not self.uniform_stride, "requested random access indices, but is in uniform stride mode"
        if traj in self.traj_keys:
            return self.ra_indices_for_traj_dict[traj]
        else:
            return np.array([])

    def ra_trajectory_length(self, traj):
        assert not self.uniform_stride, "requested random access trajectory length, but is in uniform stride mode"
        return int(self.trajectory_lengths[np.where(self.traj_keys == traj)]) if traj in self.traj_keys else 0

    @staticmethod
    def is_uniform_stride(stride):
        return not isinstance(stride, np.ndarray)

    def is_stride_sorted(self):
        if not self.uniform_stride:
            stride_traj_keys = self.stride[:, 0]
            if not all(np.diff(stride_traj_keys) >= 0):
                # traj keys were not sorted
                return False
            for idx in self.traj_keys:
                if not all(np.diff(self.stride[stride_traj_keys == idx][:, 1]) >= 0):
                    # traj indices were not sorted
                    return False
        return True


class DataSourceIterator(metaclass=ABCMeta):
    """
    Abstract class for any data source iterator.
    """
    def __init__(self, data_source, skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        self._data_source = data_source
        self.state = IteratorState(skip=skip, chunk=chunk,
                                   return_trajindex=return_trajindex,
                                   ntraj=self.number_of_trajectories(),
                                   cols=cols)
        self.__init_stride(stride)
        self._last_chunk_in_traj = False
        # the currently selected itraj, used as a guard to avoid opening the same file multiple times.
        self._selected_itraj = -1
        self._skip_unselected_or_too_short_trajs()
        super(DataSourceIterator, self).__init__()

    def __init_stride(self, stride):
        self.state.stride = stride
        if isinstance(stride, np.ndarray):
            # shift frame indices by skip
            self.state.stride[:, 1] += self.state.skip
            keys = stride[:, 0]
            if keys.max() >= self.number_of_trajectories():
                raise ValueError("provided too large trajectory index in stride argument (given max index: %s, "
                                 "allowed: %s)" % (keys.max(), self.number_of_trajectories() - 1))
            self.state.traj_keys, self.state.trajectory_lengths = np.unique(keys, return_counts=True)
            self.state.ra_indices_for_traj_dict = {}
            for traj in self.state.traj_keys:
                self.state.ra_indices_for_traj_dict[traj] = self.state.stride[self.state.stride[:, 0] == traj][:, 1]
        else:
            self.state.traj_keys = None
        self.state.uniform_stride = IteratorState.is_uniform_stride(stride)
        if not IteratorState.is_uniform_stride(stride):
            if not self.state.is_stride_sorted():
                raise ValueError("Only sorted arrays allowed for iterator pseudo random access")
            # skip trajs which are not included in stride
            while self.state.itraj not in self.state.traj_keys and self.state.itraj < self._data_source.ntraj:
                self.state.itraj += 1

    def ra_indices_for_traj(self, traj):
        """
        Gives the indices for a trajectory file index (without changing the order within the trajectory itself).
        :param traj: a trajectory file index
        :return: a Nx1 - np.array of the indices corresponding to the trajectory index
        """
        return self.state.ra_indices_for_traj(traj)

    def ra_trajectory_length(self, traj):
        return self.state.ra_trajectory_length(traj)

    def is_stride_sorted(self):
        return self.state.is_stride_sorted()

    @property
    def n_chunks(self):
        """ rough estimate of how many chunks will be processed """
        return self._data_source.n_chunks(self.chunksize, stride=self.stride, skip=self.skip)

    def number_of_trajectories(self):
        return self._data_source.number_of_trajectories()

    def trajectory_length(self, itraj=None):
        if itraj is None:
            itraj = self.current_trajindex
        return self._data_source.trajectory_length(itraj, self.stride, self.skip)

    def trajectory_lengths(self):
        return self._data_source.trajectory_lengths(self.stride, self.skip)

    def n_frames_total(self):
        return self._data_source.n_frames_total(stride=self.stride, skip=self.skip)

    @abstractmethod
    def close(self):
        """ closes the reader"""
        raise NotImplementedError()

    @staticmethod
    def _select_file_guard(datasource_method):
        """ in case we call _select_file multiple times with the same value, we do not want to reopen file handles."""
        from functools import wraps
        @wraps(datasource_method)
        def wrapper(self, itraj):
            # itraj already selected, we're done.
            if itraj == self._selected_itraj:
                return
            datasource_method(self, itraj)
            self._itraj = self._selected_itraj = itraj
        return wrapper

    @abstractmethod
    def _select_file(self, itraj):
        """ opens the next file defined by itraj.

        Notes
        -----
        Should also set self._itraj and self._selected_itraj, if the opening was successful.

        Parameters
        ----------
        itraj : int
            index of trajectory to open.
        """
        raise NotImplementedError()

    def reset(self):
        """
        Method allowing to reset the iterator so that it can iteration from beginning on again.
        """
        self._select_file(0)

    @property
    def pos(self):
        """
        Gives the current position in the current trajectory. The position is always referring to the index of the
        first frame that got yielded.

        Returns
        -------
        int
            The current iterator's position in the current trajectory.
        """
        return self.state.pos

    @property
    def current_trajindex(self):
        """
        Gives the current iterator's trajectory index.
        Returns
        -------
        int
            The current iterator's trajectory index.
        """
        return self.state.current_itraj

    @property
    def use_cols(self):
        return self.state.cols

    @property
    def skip(self):
        """
        Returns the skip value, i.e., the number of frames that are being omitted at the beginning of each
        trajectory.
        Returns
        -------
        int
            The skip value.
        """
        return self.state.skip

    @property
    def _t(self):
        """
        Reader-internal property that tracks the upcoming iterator position. Should not be used within iterator loop.
        Returns
        -------
        int
            The upcoming iterator position.
        """
        return self.state.t

    @_t.setter
    def _t(self, value):
        """
        Reader-internal property that tracks the upcoming iterator position.
        Parameters
        ----------
        value : int
            The upcoming iterator position.
        """
        self.state.t = value

    @property
    def _t_abs(self):
        """ absolute time counter, includes skip and stride. """
        return self.skip + self._t * self.stride

    @property
    def _itraj(self):
        """
        Reader-internal property that tracks the upcoming trajectory index. Should not be used within iterator loop.
        Returns
        -------
        int
            The upcoming trajectory index.
        """
        return self.state.itraj

    @_itraj.setter
    def _itraj(self, value):
        """
        Reader-internal property that tracks the upcoming trajectory index. Should not be used within iterator loop.
        Parameters
        ----------
        value : int
            The upcoming trajectory index.
        """
        if value != self._selected_itraj:
            self.state.itraj = value
            # TODO: this side effect is unexpected.
            self.state.t = 0

    def _skip_unselected_or_too_short_trajs(self):
        value = self._itraj
        if not self.uniform_stride:
            # skip trajs not included in random access stride
            while (value not in self.traj_keys or self._t >= self.ra_trajectory_length(value)) \
                    and value < self.state.ntraj:
                value += 1
                self._t = 0
        else:
            while value < self.state.ntraj and self._t >= self.trajectory_length(value):
                value += 1
                self._t = 0
        if value != self._itraj:
            self._itraj = value
            self.state.pos_adv = 0

    @skip.setter
    def skip(self, value):
        """
        Sets the skip parameter. This can be used to skip the first n frames of the next trajectory in the iterator.
        Parameters
        ----------
        value : int
            The new skip parameter.
        """
        self.state.skip = value

    @property
    def chunksize(self):
        """
        The current chunksize of the iterator. Can be changed dynamically during iteration.
        Returns
        -------
        int
            The current chunksize of the iterator.
        """
        return self.state.chunk

    @chunksize.setter
    def chunksize(self, value):
        """
        Sets the current chunksize of the iterator. Can be changed dynamically during iteration.
        Parameters
        ----------
        value : int
            The chunksize of the iterator. Required to be non-negative.
        """
        if not value >= 0:
            raise ValueError("chunksize has to be non-negative")
        self.state.chunk = value

    @property
    def stride(self):
        """
        Gives the current stride parameter.
        Returns
        -------
        int
            The current stride parameter.
        """
        return self.state.stride

    @stride.setter
    def stride(self, value):
        """
        Sets the stride parameter.
        Parameters
        ----------
        value : int
            The new stride parameter.
        """
        self.__init_stride(value)

    @property
    def return_traj_index(self):
        """
        Property that gives information whether the trajectory index gets returned during the iteration.
        Returns
        -------
        bool
            True if the trajectory index should be returned, otherwise False.
        """
        return self.state.return_trajindex

    @property
    def traj_keys(self):
        """
        Random access property returning the trajectory indices that were handed in.
        Returns
        -------
        list
            Trajectories that are used in random access.
        """
        return self.state.traj_keys

    @property
    def uniform_stride(self):
        """
        Boolean property that tells if the stride argument was integral (i.e., uniform stride) or a random access
        dictionary.
        Returns
        -------
        bool
            True if the stride argument was integral, otherwise False.
        """
        return self.state.uniform_stride

    @return_traj_index.setter
    def return_traj_index(self, value):
        """
        Setter for return_traj_index, determining if the trajectory index gets returned in the iteration loop.
        Parameters
        ----------
        value : bool
            True if it should be returned, otherwise False
        """
        self.state.return_trajindex = value

    @staticmethod
    def is_uniform_stride(stride):
        return IteratorState.is_uniform_stride(stride)

    @property
    def last_chunk(self):
        """
        Property returning if the current chunk is the last chunk before the iterator terminates.
        Returns
        -------
        bool
            True if the iterator terminates after the current chunk, otherwise False
        """
        return self.current_trajindex == self.number_of_trajectories() - 1 and self.last_chunk_in_traj

    @property
    def last_chunk_in_traj(self):
        """
        Property returning if the current chunk is the last chunk before the iterator terminates or the next trajectory.
        Returns
        -------
        bool
            True if the next chunk either belongs to a new trajectory or the iterator terminates.
        """
        if self.chunksize > 0:
            return self._last_chunk_in_traj
        else:
            return True

    @abstractmethod
    def _next_chunk(self):
        raise NotImplementedError()

    def _use_cols(self, X):
        if self.use_cols is not None:
            return X[:, self.use_cols]
        return X

    def __next__(self):
        # the position is the previous advanced position
        self.state.pos = self.state.pos_adv

        # increase itraj, needed for RA stride
        # TODO: figure out why, shouldn't ctor and post iteration skipping be sufficient?
        self._skip_unselected_or_too_short_trajs()

        if self._itraj >= self.state.ntraj:  # we never want to increase this value larger than ntraj.
            self.close()
            raise StopIteration('out of files bound')
        # obtain the current trajectory index, before (potentially) incrementing it.
        self.state.current_itraj = self._itraj
        self._select_file(self._itraj)
        try:
            X = self._use_cols(self._next_chunk())
            self._t += len(X)
        except StopIteration as e:
            self._last_chunk_in_traj = True
            raise
        # now increase itraj if needed, remember last time position, because the skip method resets _t
        self._skip_unselected_or_too_short_trajs()

        if self.state.current_itraj != self._itraj:
            self._last_chunk_in_traj = True
        else:
            if self.uniform_stride:
                length = self._data_source.trajectory_length(itraj=self.state.current_itraj,
                                                             stride=self.stride, skip=self.skip)
            else:
                length = self.ra_trajectory_length(self.state.current_itraj)
            self._last_chunk_in_traj = self.pos >= length

        if config.coordinates_check_output:
            finite = self._data_source._chunk_finite(X)
            if not np.all(finite):
                # determine position
                frames = np.where(np.logical_not(finite))
                msg = 'Found invalid values in chunk in trajectory index {itraj} at chunk [{start}, {stop}] ' \
                      'within frames {frames}.'.format(itraj=self.current_trajindex, start=self._t,
                                                       stop=self._t + len(X), frames=frames)
                raise InvalidDataInStreamException(msg)

        self.state.pos_adv = self._t
        if self.return_traj_index:
            return self.state.current_itraj, X
        return X

    next = __next__

    def __iter__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __str__(self):
        return "[{name} itraj={itraj}, traj_len={traj_len}, curr_traj_ind={cur_ind}, chunk={chunk}," \
               " stride={stride}, skip={skip}, t={t}, pos={pos}]".format(
            name=self.__class__.__name__,
            chunk=self.chunksize,
            stride=self.stride,
            skip=self.skip,
            t=self._t,
            itraj=self._itraj,
            cur_ind=self.current_trajindex,
            pos=self.pos,
            traj_len=self.trajectory_length()
        )


class EncapsulatedIterator(DataSourceIterator):
    """
    Parameters
    ----------
    data_source
    iterator
    transform_function
    skip
    chunk
    stride
    return_trajindex
    cols
    """
    def __init__(self, data_source, iterator=None, transform_function=None,
                 skip=0, chunk=0, stride=1, return_trajindex=False, cols=None):
        super(EncapsulatedIterator, self).__init__(data_source=data_source, skip=skip, chunk=chunk,
                                                   stride=stride, return_trajindex=return_trajindex, cols=cols)
        self._it = iterator
        self.transform_function = transform_function
        self._select_file(0)
        assert self._it is not None
        # map the reference of the real used iterator to this instance to avoid overriding every attribute.
        if hasattr(self._it, 'state'):
            self.state = self._it.state

    @DataSourceIterator.chunksize.setter
    def chunksize(self, value):
        self.state.chunk = value
        if hasattr(self._it, 'chunksize'):
            self._it.chunksize = value

    @DataSourceIterator.skip.setter
    def skip(self, value):
        self.state.skip = value
        if hasattr(self._it, 'skip'):
            self._it.skip = value

    @property
    def transform_function(self):
        return self._transform_function

    @transform_function.setter
    def transform_function(self, value):
        if value is not None and not callable(value):
            raise ValueError('transform function has to be callable. Given value: {}'.format(value))
        self._transform_function = value

    @DataSourceIterator._select_file_guard
    def _select_file(self, itraj):
        self._it._select_file(itraj)

    def close(self):
        if self._it is not None and hasattr(self._it, 'close'):
            self._it.close()

    def _next_chunk(self):
        if hasattr(self._it, '_next_chunk'):
            x = self._it._next_chunk()
        else:
            x = next(self._it)
        # We discard the trajectory index here for transformation
        if self.transform_function is not None:
            x = self.transform_function(x)
        return x


class InvalidDataInStreamException(Exception):
    """Data stream contained NaN or (+/-) infinity"""
