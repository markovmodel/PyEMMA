'''
Created on 13.03.2015

@author: marscher
'''
import numpy as np
import warnings

from mdtraj.utils.validation import cast_indices
from mdtraj.core.trajectory import load, Trajectory, _parse_topology
from mdtraj.formats.hdf5 import HDF5TrajectoryFile
from mdtraj.utils.unit import in_units_of
from mdtraj.formats.lh5 import LH5TrajectoryFile
from mdtraj.formats import DCDTrajectoryFile
from mdtraj.formats import XTCTrajectoryFile

from pyemma.util.log import getLogger

log = getLogger('patches')


def iterload(filename, chunk=100, **kwargs):
    """An iterator over a trajectory from one or more files on disk, in fragments

    This may be more memory efficient than loading an entire trajectory at
    once

    Parameters
    ----------
    filename : str
        Path to the trajectory file on disk
    chunk : int
        Number of frames to load at once from disk per iteration.  If 0, load all.

    Other Parameters
    ----------------
    top : {str, Trajectory, Topology}
        Most trajectory formats do not contain topology information. Pass in
        either the path to a RCSB PDB file, a trajectory, or a topology to
        supply this information. This option is not required for the .h5, .lh5,
        and .pdb formats, which already contain topology information.
    stride : int, default=None
        Only read every stride-th frame.
    atom_indices : array_like, optional
        If not none, then read only a subset of the atoms coordinates from the
        file. This may be slightly slower than the standard read because it
        requires an extra copy, but will save memory.

    See Also
    --------
    load, load_frame

    Examples
    --------
    >>> import mdtraj as md
    >>> for chunk in md.iterload('output.xtc', top='topology.pdb')
    ...    print chunk
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    """
    stride = kwargs.get('stride', 1)
    atom_indices = cast_indices(kwargs.get('atom_indices', None))
    if chunk % stride != 0 and filename.endswith('.dcd'):
        raise ValueError('Stride must be a divisor of chunk. stride=%d does not go '
                         'evenly into chunk=%d' % (stride, chunk))
    if chunk == 0:
        yield load(filename, **kwargs)
    else:  # If chunk was 0 then we want to avoid filetype-specific code in case of undefined behavior in various file parsers.
        skip = kwargs.get('skip', 0)
        if filename.endswith('.h5'):
            if 'top' in kwargs:
                warnings.warn('top= kwarg ignored since file contains topology information')

            with HDF5TrajectoryFile(filename) as f:
                if skip > 0:
                    xyz, _, _, _ = f.read(skip, atom_indices=atom_indices)
                    if len(xyz) == 0:
                        raise StopIteration()
                if atom_indices is None:
                    topology = f.topology
                else:
                    topology = f.topology.subset(atom_indices)

                while True:
                    data = f.read(chunk*stride, stride=stride, atom_indices=atom_indices)
                    if data == []:
                        raise StopIteration()
                    in_units_of(data.coordinates, f.distance_unit, Trajectory._distance_unit, inplace=True)
                    in_units_of(data.cell_lengths, f.distance_unit, Trajectory._distance_unit, inplace=True)
                    yield Trajectory(xyz=data.coordinates, topology=topology,
                                     time=data.time, unitcell_lengths=data.cell_lengths,
                                     unitcell_angles=data.cell_angles)

        if filename.endswith('.lh5'):
            if 'top' in kwargs:
                warnings.warn('top= kwarg ignored since file contains topology information')
            with LH5TrajectoryFile(filename) as f:
                if atom_indices is None:
                    topology = f.topology
                else:
                    topology = f.topology.subset(atom_indices)

                ptr = 0
                if skip > 0:
                    xyz, _, _, _ = f.read(skip, atom_indices=atom_indices)
                    if len(xyz) == 0:
                        raise StopIteration()
                while True:
                    xyz = f.read(chunk*stride, stride=stride, atom_indices=atom_indices)
                    if len(xyz) == 0:
                        raise StopIteration()
                    in_units_of(xyz, f.distance_unit, Trajectory._distance_unit, inplace=True)
                    time = np.arange(ptr, ptr+len(xyz)*stride, stride)
                    ptr += len(xyz)*stride
                    yield Trajectory(xyz=xyz, topology=topology, time=time)

        elif filename.endswith('.xtc'):
            topology = _parse_topology(kwargs.get('top', None))
            with XTCTrajectoryFile(filename) as f:
                if skip > 0:
                    xyz, _, _, _ = f.read(skip)
                    if len(xyz) == 0:
                        raise StopIteration()
                while True:
                    xyz, time, step, box = f.read(chunk*stride, stride=stride, atom_indices=atom_indices)
                    if len(xyz) == 0:
                        raise StopIteration()
                    in_units_of(xyz, f.distance_unit, Trajectory._distance_unit, inplace=True)
                    in_units_of(box, f.distance_unit, Trajectory._distance_unit, inplace=True)
                    trajectory = Trajectory(xyz=xyz, topology=topology, time=time)
                    trajectory.unitcell_vectors = box
                    yield trajectory

        elif filename.endswith('.dcd'):
            topology = _parse_topology(kwargs.get('top', None))
            with DCDTrajectoryFile(filename) as f:
                ptr = 0
                if skip > 0:
                    xyz, _, _ = f.read(skip, atom_indices=atom_indices)
                    if len(xyz) == 0:
                        raise StopIteration()
                while True:
                    # for reasons that I have not investigated, dcdtrajectory file chunk and stride
                    # together work like this method, but HDF5/XTC do not.
                    xyz, box_length, box_angle = f.read(chunk, stride=stride, atom_indices=atom_indices)
                    if len(xyz) == 0:
                        raise StopIteration()
                    in_units_of(xyz, f.distance_unit, Trajectory._distance_unit, inplace=True)
                    in_units_of(box_length, f.distance_unit, Trajectory._distance_unit, inplace=True)
                    time = np.arange(ptr, ptr+len(xyz)*stride, stride)
                    ptr += len(xyz)*stride
                    yield Trajectory(xyz=xyz, topology=topology, time=time, unitcell_lengths=box_length,
                                     unitcell_angles=box_angle)

        else:
            log.critical("loading complete traj into mem! This might no be desired.")
            t = load(filename, **kwargs)
            for i in range(skip, len(t), chunk):
                yield t[i:i+chunk]
