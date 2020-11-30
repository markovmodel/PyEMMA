
# This file is part of PyEMMA.
#
# Copyright (c) 2014-2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

from pickle import Pickler, Unpickler, UnpicklingError

import numpy as np
import logging

try:
    import tables
except:
    pass

logger = logging.getLogger(__name__)

__author__ = 'marscher'


def _blosc_opts(complevel=9, complib='blosc:lz4', shuffle=True):
    shuffle = 2 if shuffle == 'bit' else 1 if shuffle else 0
    compressors = ['blosclz', 'lz4', 'lz4hc', 'snappy', 'zlib', 'zstd']
    complib = ['blosc:' + c for c in compressors].index(complib)
    args = {
        'compression': 32001,
        'compression_opts': (0, 0, 0, 0, complevel, shuffle, complib)
    }
    if shuffle:
        args['shuffle'] = False
    return args


def _check_blosc_avail():
    import tempfile, h5py
    blosc_opts = _blosc_opts()
    fid, name = tempfile.mkstemp()
    try:
        with h5py.File(name, mode="w") as h5f:
            try:
                h5f.create_dataset('test', shape=(1,1), **blosc_opts)
            except ValueError as ve:
                if 'Unknown compression filter' in str(ve):
                    import warnings
                    warnings.warn('BLOSC compression filter unavailable. '
                                  'Your resulting file may be large and not optimal to process.')
                    return {}
                else:  # unknown exception
                    raise
            else:
                return blosc_opts
    finally:
        try:
            import os
            os.unlink(name)
        except:
            pass


# we cache this during runtime
_DEFAULT_BLOSC_OPTIONS = _check_blosc_avail()


class HDF5PersistentPickler(Pickler):
    # stores numpy arrays during pickling in given hdf5 group.
    def __init__(self, group, file):
        super().__init__(file=file, protocol=4)
        self.group = group
        self._seen_ids = set()

    def dump(self, *args, **kwargs):
        # we temporarily patch mdtraj.Topology to save state to numpy array
        from unittest import mock
        from pyemma._base.serialization.mdtraj_helpers import getstate
        with mock.patch('mdtraj.Topology.__getstate__', getstate, create=True):
            super(HDF5PersistentPickler, self).dump(*args, **kwargs)

    def _store(self, array):
        id_ = id(array)
        key = str(id_)
        # this actually makes no sense to check it here, however it is needed,
        # since there seems to be some race condition in h5py...
        if key in self.group:
            assert id_ in self._seen_ids
            return id_
        self._seen_ids.add(id_)
        self.group.create_dataset(name=key, data=array,
                                  chunks=True, **_DEFAULT_BLOSC_OPTIONS)
        return id_

    def persistent_id(self, obj):
        if (isinstance(obj, np.ndarray) and obj.dtype != np.object_
                and id(obj) not in self._seen_ids):
            # do not store empty arrays in hdf (more overhead)
            if not len(obj):
                return None
            array_id = self._store(obj)
            return 'np_array', array_id

        return None


class HDF5PersistentUnpickler(Unpickler):
    __allowed_packages = ('builtin',
                          'pyemma',
                          'mdtraj',
                          'numpy',
                          'scipy',
                          'bhmm',
                          )

    def __init__(self, group, file):
        super().__init__(file=file)
        self.group = group

    def persistent_load(self, pid):
        # This method is invoked whenever a persistent ID is encountered.
        # Here, pid is the type and the dataset id.
        type_tag, key_id = pid
        if type_tag == "np_array":
            return self.group[str(key_id)][:]
        else:
            # Always raises an error if you cannot return the correct object.
            # Otherwise, the unpickler will think None is the object referenced
            # by the persistent ID.
            raise UnpicklingError("unsupported persistent object")

    def load(self, *args, **kwargs):
        # we temporarily patch mdtraj.Topology to load state from numpy array
        from unittest import mock
        from pyemma._base.serialization.mdtraj_helpers import setstate
        with mock.patch('mdtraj.Topology.__setstate__', setstate, create=True):
            return super(HDF5PersistentUnpickler, self).load(*args, **kwargs)

    @staticmethod
    def __check_allowed(module):
        # check if we are allowed to unpickle from these modules.
        i = module.find('.')
        if i > 0:
            package = module[:i]
        else:
            package = module
        if package not in HDF5PersistentUnpickler.__allowed_packages:
            raise UnpicklingError('{mod} not allowed to unpickle'.format(mod=module))

    def find_class(self, module, name):
        self.__check_allowed(module)
        from .util import class_rename_registry
        new_class = class_rename_registry.find_replacement_for_old('{}.{}'.format(module, name))
        if new_class:
            return new_class
        return super(HDF5PersistentUnpickler, self).find_class(module, name)
