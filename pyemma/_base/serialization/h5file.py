import logging
import numpy as np
from io import BytesIO
from .pickle_extensions import HDF5PersistentPickler


logger = logging.getLogger(__name__)


class H5Wrapper(object):
    stored_attributes = ('created',
                         'created_readable',
                         'class_str',
                         'class_repr',
                         'saved_streaming_chain',
                         'model',
                         'pyemma_version',
                         'digest',
                         )

    def __init__(self, file_name: str, model_name=None, mode=None):
        import h5py
        self._file = h5py.File(file_name, mode=mode)
        self._parent = self._file.require_group('pyemma')
        self._current_model_group = model_name

    @property
    def _current_model_group(self):
        return self._current_model_name

    @_current_model_group.setter
    def _current_model_group(self, model_name: str):
        if model_name is None:
            self._current_model_name = None
        else:
            if model_name in self._parent and \
                    not all(attr in self._parent[model_name].attrs for attr in H5Wrapper.stored_attributes):
                raise ValueError('already saved model does not contain desired attributes: {}. Contains only: {}'
                                 .format(H5Wrapper.stored_attributes, self._parent[model_name].attrs))
            self._current_model_name = self._parent.require_group(model_name)

    @_current_model_group.deleter
    def _current_model_group(self):
        # delete model from file
        if self._current_model_group is None:
            raise AttributeError('can not delete current group, because it is not set.')
        del self._parent[self._current_model_group.name]
        self._current_model_name = None

    @property
    def model(self):
        # restore pickled object.
        g = self._current_model_group
        inp = g.attrs['model']
        from .pickle_extensions import HDF5PersistentUnpickler
        from io import BytesIO
        file = BytesIO(inp)
        # validate hash.
        self._hash(g.attrs, compare_to=g.attrs['digest'])
        unpickler = HDF5PersistentUnpickler(g, file=file)
        obj = unpickler.load()
        obj._restored_from_pyemma_version = g.attrs['pyemma_version']

        return obj

    def add_serializable(self, name, obj, overwrite=False, save_streaming_chain=False):
        # create new group with given name and serialize the object in it.
        from pyemma._base.serialization.serialization import SerializableMixIn
        assert isinstance(obj, SerializableMixIn)

        from pyemma import version
        import time

        if name in self._parent:
            if overwrite:
                logger.info('overwriting model "%s" in file %s', name, self._file.name)
                self._current_model_group = name
                del self._current_model_group
            else:
                raise RuntimeError('model "{name}" already exists. Either use overwrite=True,'
                                   ' or use a different name/file.')

        old_flag = getattr(obj, '_save_data_producer', None)
        if old_flag is not None:
            obj._save_data_producer = save_streaming_chain
            assert obj._save_data_producer == save_streaming_chain

        try:
            self._current_model_group = name
            g = self._current_model_group
            g.attrs['created'] = time.time()
            g.attrs['created_readable'] = time.asctime()
            g.attrs['class_str'] = str(obj)
            g.attrs['class_repr'] = repr(obj)
            g.attrs['saved_streaming_chain'] = save_streaming_chain
            # store the current software version
            g.attrs['pyemma_version'] = version

            # now encode the object (this will write all numpy arrays to current group).
            file = BytesIO()
            pickler = HDF5PersistentPickler(g, file=file)
            pickler.dump(obj)
            file.seek(0)
            flat = file.read()
            # attach the pickle byte string to the H5 file.
            g.attrs['model'] = np.void(flat)
            # integrity check
            g.attrs['digest'] = H5Wrapper._hash(g.attrs)
        finally:
            # restore old state.
            if old_flag is not None:
                obj._save_data_producer = old_flag

    @property
    def models_descriptive(self):
        """ list all stored models in given file.

        Parameters
        ----------
        file_name: str
            path to file to list models for

        Returns
        -------
        dict: {model_name: {'repr' : 'string representation, 'created': 'human readable date', ...}

        """
        f = self._parent
        return {name: {a: f[name].attrs[a]
                       for a in H5Wrapper.stored_attributes if a != 'model'}
                for name in f.keys()}

    @staticmethod
    def _hash(attributes, compare_to=None):
        # hashes the attributes in the hdf5 file (also binary data), to make it harder to manipulate them.
        import hashlib
        digest = hashlib.sha256()
        for attr in H5Wrapper.stored_attributes:
            if attr == 'digest':
                continue
            value = attributes[attr]
            if attr == 'model':  # do not convert to ascii.
                pair = value
            else:
                # TODO: needed?
                pair = '{}={}'.format(attr, value).encode('ascii')
            digest.update(pair)
        hex = digest.hexdigest()
        if compare_to is not None and hex != compare_to:
            from pyemma._base.serialization.serialization import IntegrityError
            raise IntegrityError('mismatch:{} !=\n{}'.format(digest, compare_to))
        return digest.hexdigest()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.close()
