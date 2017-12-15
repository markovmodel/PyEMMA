import numpy as np


def topology_to_numpy(top):
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


def topology_from_numpy(atoms, bonds=None):
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
