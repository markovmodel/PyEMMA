import pickle
from pickle import Pickler, Unpickler

import mdtraj
import numpy as np
from pyemma._base.serialization.mdtraj_helpers import topology_to_numpy, topology_from_numpy
from pyemma._base.serialization.util import class_rename_registry


class HDF5PersistentPickler(Pickler):
    # stores numpy arrays during pickling in given hdf5 group.
    def __init__(self, group, file):
        super().__init__(file=file, protocol=4)
        self.group = group
        from itertools import count
        self.next_array_id = count(0)

    def _store(self, array):
        array_id = next(self.next_array_id)
        self.group.create_dataset(name=str(array_id), data=array,
                                  chunks=True, compression='gzip', compression_opts=4, shuffle=True)
        return array_id

    def persistent_id(self, obj):
        if isinstance(obj, np.ndarray) and obj.dtype != np.object_:
            array_id = self._store(obj)
            return 'np_array', array_id

        if isinstance(obj, mdtraj.Topology):
            atoms, bonds = topology_to_numpy(obj)
            atom_i = self._store(atoms)
            bond_i = self._store(bonds)
            return "md/Topology", (atom_i, bond_i)

        return None


class HDF5PersistentUnpickler(Unpickler):
    __allowed_packages = ('builtin',
                          'pyemma',
                          'mdtraj',
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
        elif type_tag == 'md/Topology':
            atoms = self.group[str(key_id[0])][:]
            bonds = self.group[str(key_id[1])][:]
            return topology_from_numpy(atoms, bonds)
        else:
            # Always raises an error if you cannot return the correct object.
            # Otherwise, the unpickler will think None is the object referenced
            # by the persistent ID.
            raise pickle.UnpicklingError("unsupported persistent object")

    @staticmethod
    def _check_allowed(module):
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
