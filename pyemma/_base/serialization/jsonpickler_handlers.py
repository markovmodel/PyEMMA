"""
This module contains custom serialization handlers for jsonpickle to flatten and restore given types.

@author: Martin K. Scherer
"""
import numpy as np

from jsonpickle import handlers


def register_ndarray_handler():
    H5BackendLinkageHandler.handles(np.ndarray)

    for t in NumpyExtractedDtypeHandler.np_dtypes:
        NumpyExtractedDtypeHandler.handles(t)

def register_all_handlers():
    register_ndarray_handler()
    import mdtraj
    TopologyHandler.handles(mdtraj.Topology)


class H5BackendLinkageHandler(handlers.BaseHandler):
    """ stores NumPy arrays in the backing hdf5 file contained in the context """
    def __init__(self, context):
        if not hasattr(context, 'h5_file'):
            raise ValueError('the given un/-pickler has to contain a hdf5 file reference.')

        from jsonpickle.pickler import Pickler
        if isinstance(self, Pickler) and not hasattr(context, 'next_array_id'):
            raise ValueError('the given pickler has to contain an array id provider')
        super(H5BackendLinkageHandler, self).__init__(context=context)

    @property
    def file(self):
        # obtain the current file handler
        return self.context.h5_file

    def next_array_id(self):
        n = lambda: str(next(self.context.next_array_id))
        res = n()
        while res in self.file:
            res = n()
        return res

    def flatten(self, obj, data):
        if obj.dtype == np.object_:
            value = [self.context.flatten(v, reset=False) for v in obj]
            data['values'] = value
        else:
            array_id = self.next_array_id()
            self.file.create_dataset(name=array_id, data=obj,
                                     chunks=True, compression='gzip', compression_opts=4, shuffle=True)
            data['array_ref'] = array_id
        return data

    def restore(self, obj):
        if 'array_ref' in obj:
            array_ref = obj['array_ref']
            # it is important to return a copy here, because h5 only creates views to the data.
            ds = self.file[array_ref]
            return ds[:]
        else:
            values = obj['values']
            result = np.empty(len(values), dtype=object)
            for i, e in enumerate(values):
                result[i] = self.context.restore(e, reset=False)
            return result


class NumpyExtractedDtypeHandler(handlers.BaseHandler):
    """
    if one extracts a value from a numpy array, the resulting type is numpy.int64 etc.
    We convert these values to Python primitives right here.

    All float types up to float64 are mapped by builtin.float
    All integer (signed/unsigned) types up to int64 are mapped by builtin.int
    """
    integers = (np.bool_,
                np.int8, np.int16, np.int32, np.int64,
                np.uint8, np.uint16, np.uint32, np.uint64)
    floats__ = (np.float16, np.float32, np.float64)

    np_dtypes = integers + floats__

    def __init__(self, context):
        super(NumpyExtractedDtypeHandler, self).__init__(context=context)

    def flatten(self, obj, data):
        if isinstance(obj, self.floats__):
            data['value'] = '{:.18f}'.format(obj).rstrip('0').rstrip('.')
        elif isinstance(obj, self.integers):
            data['value'] = int(obj)
        elif isinstance(obj, np.bool_):
            data['value'] = bool(obj)
        return data

    def restore(self, obj):
        str_t = obj['py/object'].split('.')[1]
        res = getattr(np, str_t)(obj['value'])
        return res


class TopologyHandler(H5BackendLinkageHandler):

    def flatten(self, obj, data):
        atoms, bonds = self.to_dataframe(obj)
        data['atoms'] = self.context.flatten(atoms, reset=False)
        data['bonds'] = self.context.flatten(bonds, reset=False)
        return data

    def restore(self, obj):
        atoms = self.context.restore(obj['atoms'], reset=False)
        bonds = self.context.restore(obj['bonds'], reset=False)
        out = self.from_dataframe(atoms, bonds)
        return out

    @staticmethod
    def to_dataframe(top):
        """Convert this topology into a pandas dataframe

        Returns
        -------
        atoms : pandas.DataFrame
            The atoms in the topology, represented as a data frame.
        bonds : np.ndarray
            The bonds in this topology, represented as an n_bonds x 2 array
            of the indices of the atoms involved in each bond.
        """
        data = [(atom.serial, atom.name, atom.element.symbol,
                 atom.residue.resSeq, atom.residue.name,
                 atom.residue.chain.index, atom.segment_id) for atom in top.atoms]
        atoms = np.array(data,
                        dtype=[("serial", 'i4'), ("name", 'S4'), ("element", 'S3'),
                               ("resSeq", 'i4'), ("resName",'S4'), ("chainID", 'i4'), ("segmentID", 'S4')])

        bonds = np.array([(a.index, b.index) for (a, b) in top.bonds])
        return atoms, bonds

    @staticmethod
    def from_dataframe(atoms, bonds=None):
        """Create a mdtraj topology from a pandas data frame

        Parameters
        ----------
        atoms : pandas.DataFrame
            The atoms in the topology, represented as a data frame. This data
            frame should have columns "serial" (atom index), "name" (atom name),
            "element" (atom's element), "resSeq" (index of the residue)
            "resName" (name of the residue), "chainID" (index of the chain),
            and optionally "segmentID", following the same conventions
            as wwPDB 3.0 format.
        bonds : np.ndarray, shape=(n_bonds, 2), dtype=int, optional
            The bonds in the topology, represented as an n_bonds x 2 array
            of the indices of the atoms involved in each bond. Specifiying
            bonds here is optional. To create standard protein bonds, you can
            use `create_standard_bonds` to "fill in" the bonds on your newly
            created Topology object

        See Also
        --------
        create_standard_bonds
        """

        if bonds is None:
            bonds = np.zeros((0, 2))

        for col in ["name", "element", "resSeq",
                    "resName", "chainID", "serial"]:
            if col not in atoms.dtype.names:
                raise ValueError('dataframe must have column %s' % col)

        if "segmentID" not in atoms.dtype.names:
            atoms["segmentID"] = ""

        import mdtraj
        from mdtraj.core.topology import Atom
        from mdtraj.core import element as elem
        out = mdtraj.Topology()

        if not isinstance(bonds, np.ndarray):
            raise TypeError('bonds must be an instance of numpy.ndarray. '
                            'You supplied a %s' % type(bonds))

        out._atoms = [None for _ in range(len(atoms))]
        atom_index = 0

        for ci in np.unique(atoms['chainID']):
            chain_atoms = atoms[atoms['chainID'] == ci]
            c = out.add_chain()

            for ri in np.unique(chain_atoms['resSeq']):
                residue_atoms = chain_atoms[chain_atoms['resSeq'] == ri]
                rnames = residue_atoms['resName']
                residue_name = np.array(rnames)[0]
                segids = residue_atoms['segmentID']
                segment_id = np.array(segids)[0]
                if not np.all(rnames == residue_name):
                    raise ValueError('All of the atoms with residue index %d '
                                     'do not share the same residue name' % ri)
                r = out.add_residue(residue_name.decode(), c, ri, segment_id.decode())

                for atom in residue_atoms:
                    e = atom['element'].decode()
                    a = Atom(atom['name'].decode(), elem.get_by_symbol(e),
                             atom_index, r, serial=atom['serial'])
                    out._atoms[atom_index] = a
                    r._atoms.append(a)
                    atom_index += 1

        for ai1, ai2 in bonds:
            out.add_bond(out.atom(ai1), out.atom(ai2))

        out._numAtoms = out.n_atoms
        return out

