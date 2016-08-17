import tempfile
import numpy as np
import mdtraj
import pkg_resources


def get_top():
    return pkg_resources.resource_filename(__name__, 'data/test.pdb')


def create_top(n_atoms):
    from mdtraj import Topology
    t = Topology()
    chain = t.add_chain()
    res = t.add_residue("res", chain=chain)
    [t.add_atom("{}".format(n), element=None, residue=res) for n in range(n_atoms)]
    return t


def create_traj_given_xyz(xyz, top, format, directory):
    xyz = xyz.reshape(-1, top.n_atoms, 3)
    trajfile = tempfile.mktemp(suffix=format, dir=directory)

    traj = mdtraj.Trajectory(xyz=xyz, topology=top)
    traj.save(filename=trajfile)
    return trajfile

def create_traj(top=None, format='.xtc', dir=None, length=1000, start=0):
    trajfile = tempfile.mktemp(suffix=format, dir=dir)
    xyz = np.arange(start * 3 * 3, (start + length) * 3 * 3)
    xyz = xyz.reshape((-1, 3, 3))
    if top is None:
        top = get_top()

    t = mdtraj.load(top)
    t.xyz = xyz
    t.unitcell_vectors = np.array(length * [[0, 0, 1], [0, 1, 0], [1, 0, 0]]).reshape(length, 3, 3)
    t.time = np.arange(length)
    t.save(trajfile)

    return trajfile, xyz, length
