import pickle
from pickle import Pickler, Unpickler

import numpy as np
from pyemma._base.serialization.util import class_rename_registry


class HDF5PersistentPickler(Pickler):
    def __init__(self, group, file):
        super().__init__(file=file, protocol=4)
        self.group = group
        from itertools import count
        self.next_array_id = count(0)

    def persistent_id(self, obj):
        if isinstance(obj, np.ndarray) and obj.dtype != np.object_:
            array_id = next(self.next_array_id)
            self.group.create_dataset(name=str(array_id), data=obj,
                                      chunks=True, compression='gzip', compression_opts=4, shuffle=True)
            return "np_array", array_id

        return None


class HDF5PersistentUnpickler(Unpickler):
    __allowed_packages = ('builtin',
                          'pyemma',
                          'numpy')

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
            raise pickle.UnpicklingError("unsupported persistent object")

    def _check_allowed(self, module):
        # check if we are allowed to unpickle from these modules.
        i = module.find('.')
        if i > 0:
            package = module[:i]
        else:
            package = module
        if package not in HDF5PersistentUnpickler.__allowed_packages:
            raise pickle.UnpicklingError('{mod} not allowed to unpickle'.format(mod=module))

    def find_class(self, module, name):
        self._check_allowed(module)
        new_class = class_rename_registry.find_replacement_for_old('{}.{}'.format(module, name))
        if new_class:
            from importlib import import_module
            i = new_class.rfind('.')
            mod = new_class[:i]
            class_name = new_class[i+1:]
            module = import_module(mod)
            cls = getattr(module, class_name)
            return cls

        return super(HDF5PersistentUnpickler, self).find_class(module, name)
