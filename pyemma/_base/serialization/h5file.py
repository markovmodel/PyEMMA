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
        return self.__group

    @_current_model_group.setter
    def _current_model_group(self, model_name: str):
        if model_name is None:
            self.__group = None
        else:
            if model_name in self._parent and \
                    not all(attr in self._parent[model_name].attrs for attr in H5Wrapper.stored_attributes):
                raise ValueError('already saved model does not contain desired attributes: {}. Contains only: {}'
                                 .format(H5Wrapper.stored_attributes, list(self._parent[model_name].attrs)))
            self.__group = self._parent.require_group(model_name)

    @_current_model_group.deleter
    def _current_model_group(self):
        # delete model from file
        if self._current_model_group is None:
            raise AttributeError('can not delete current group, because it is not set.')
        del self._parent[self._current_model_group.name]
        self.__group = None

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

    @property
    def created(self):
        return self._current_model_group.attrs['created']

    @created.setter
    def created(self, value):
        self._current_model_group.attrs['created'] = value

    @property
    def created_readable(self):
        return self._current_model_group.attrs['created_readable']

    @created_readable.setter
    def created_readable(self, value:str):
        self._current_model_group.attrs['created_readable'] = value

    @property
    def class_str(self):
        return self._current_model_group.attrs['class_str']

    @class_str.setter
    def class_str(self, value:str):
        self._current_model_group.attrs['class_str'] = value

    @property
    def class_repr(self):
        return self._current_model_group.attrs['class_repr']

    @class_repr.setter
    def class_repr(self, value:str):
        self._current_model_group.attrs['class_repr'] = value

    @property
    def save_streaming_chain(self):
        return self._current_model_group.attrs['saved_streaming_chain']

    @save_streaming_chain.setter
    def save_streaming_chain(self, value:bool):
        self._current_model_group.attrs['saved_streaming_chain'] = value

    @property
    def pyemma_version(self):
        return self._current_model_group.attrs['pyemma_version']

    @pyemma_version.setter
    def pyemma_version(self, value:str):
        self._current_model_group.attrs['pyemma_version'] = value

    def _set_group(self, name, overwrite=False):
        if name in self._parent:
            if overwrite:
                logger.info('overwriting model "%s" in file %s', name, self._file.name)
                self._current_model_group = 'latest'
                del self._current_model_group
            else:
                raise RuntimeError('model "{name}" already exists. Either use overwrite=True,'
                                   ' or use a different name/file.')
        self._current_model_group = name

    def add_serializable(self, name, obj, overwrite=False, save_streaming_chain=False):
        # create new group with given name and serialize the object in it.
        from pyemma._base.serialization.serialization import SerializableMixIn
        assert isinstance(obj, SerializableMixIn)

        # save data producer chain?
        old_flag = getattr(obj, '_save_data_producer', None)
        if old_flag is not None:
            obj._save_data_producer = save_streaming_chain
            assert obj._save_data_producer == save_streaming_chain

        try:
            self._set_group(name, overwrite)
            # store attributes
            self._save_attributes(obj)
            # additionally we store, whether the pipeline has been saved.
            self.save_streaming_chain = save_streaming_chain

            # now encode the object (this will write all numpy arrays to current group).
            self._pickle_and_attach_object(obj)
        finally:
            # restore old state.
            if old_flag is not None:
                obj._save_data_producer = old_flag

    def add_object(self, name, obj, overwrite=False):
        self._set_group(name, overwrite)
        self._save_attributes(obj)
        self._pickle_and_attach_object(obj)

    def _save_attributes(self, obj):
        from pyemma import version
        import time
        self.created = time.time()
        self.created_readable = time.asctime()
        self.class_str = str(obj)
        self.class_repr = repr(obj)
        # store the current software version
        self.pyemma_version = version

    def _pickle_and_attach_object(self, obj):
        # now encode the object (this will write all numpy arrays to current group).
        file = BytesIO()
        pickler = HDF5PersistentPickler(self._current_model_group, file=file)
        pickler.dump(obj)
        file.seek(0)
        flat = file.read()
        # attach the pickle byte string to the H5 file.
        attrs = self._current_model_group.attrs
        attrs['model'] = np.void(flat)
        # integrity check
        attrs['digest'] = H5Wrapper._hash(attrs)

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
            if attr == 'digest' or attr == 'saved_streaming_chain':
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
