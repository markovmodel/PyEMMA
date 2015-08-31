
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


'''
Created on 13.03.2015

@author: marscher
'''

from __future__ import absolute_import
import numpy as np
from mdtraj.utils.validation import cast_indices
from mdtraj.core.trajectory import load, _parse_topology, _TOPOLOGY_EXTS, _get_extension, open

from itertools import groupby
from operator import itemgetter

from pyemma.coordinates.data.util.reader_utils import copy_traj_attributes, preallocate_empty_trajectory
from six.moves import map
from six.moves import range


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
    >>> for chunk in md.iterload('output.xtc', top='topology.pdb') # doctest: +SKIP
    ...     print chunk # doctest: +SKIP

    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>
    <mdtraj.Trajectory with 100 frames, 423 atoms at 0x110740a90>

    """
    stride = kwargs.pop('stride', 1)
    atom_indices = cast_indices(kwargs.pop('atom_indices', None))
    top = kwargs.pop('top', None)
    skip = kwargs.pop('skip', 0)

    extension = _get_extension(filename)
    if extension not in _TOPOLOGY_EXTS:
        topology = _parse_topology(top)
    else:
        topology = top

    if chunk == 0:
        # If chunk was 0 then we want to avoid filetype-specific code
        # in case of undefined behavior in various file parsers.
        # TODO: this will first apply stride, then skip!
        if extension not in _TOPOLOGY_EXTS:
            kwargs['top'] = top
        yield load(filename, **kwargs)[skip:]
    elif extension in ('.pdb', '.pdb.gz'):
        # the PDBTrajectortFile class doesn't follow the standard API. Fixing it
        # to support iterload could be worthwhile, but requires a deep refactor.
        t = load(filename, stride=stride, atom_indices=atom_indices)
        for i in range(0, len(t), chunk):
            yield t[i:i+chunk]

    elif isinstance(stride, np.ndarray):
        with (lambda x: open(x, n_atoms=topology.n_atoms)
              if extension in ('.crd', '.mdcrd')
              else open(filename))(filename) as f:
            x_prev = 0
            curr_size = 0
            traj = []
            leftovers = []
            for k, g in groupby(enumerate(stride), lambda a: a[0] - a[1]):
                grouped_stride = list(map(itemgetter(1), g))
                seek_offset = (1 if x_prev != 0 else 0)
                seek_to = grouped_stride[0] - x_prev - seek_offset
                f.seek(seek_to, whence=1)
                x_prev = grouped_stride[-1]
                group_size = len(grouped_stride)
                if curr_size + group_size > chunk:
                    leftovers = grouped_stride
                else:
                    local_traj = _get_local_traj_object(atom_indices, extension, f, group_size, topology, **kwargs)
                    traj.append(local_traj)
                    curr_size += len(grouped_stride)
                if curr_size == chunk:
                    yield _efficient_traj_join(traj)
                    curr_size = 0
                    traj = []
                while leftovers:
                    local_chunk = leftovers[:min(chunk, len(leftovers))]
                    local_traj = _get_local_traj_object(atom_indices, extension, f, len(local_chunk), topology, **kwargs)
                    traj.append(local_traj)
                    leftovers = leftovers[min(chunk, len(leftovers)):]
                    curr_size += len(local_chunk)
                    if curr_size == chunk:
                        yield _efficient_traj_join(traj)
                        curr_size = 0
                        traj = []
            if traj:
                yield _efficient_traj_join(traj)
            raise StopIteration()

    else:
        with (lambda x: open(x, n_atoms=topology.n_atoms)
              if extension in ('.crd', '.mdcrd')
              else open(filename))(filename) as f:
            if skip > 0:
                f.seek(skip)
            while True:
                if extension not in _TOPOLOGY_EXTS:
                    traj = f.read_as_traj(topology, n_frames=chunk*stride, stride=stride, atom_indices=atom_indices, **kwargs)
                else:
                    traj = f.read_as_traj(n_frames=chunk*stride, stride=stride, atom_indices=atom_indices, **kwargs)

                if len(traj) == 0:
                    raise StopIteration()

                yield traj


def _get_local_traj_object(atom_indices, extension, f, n_frames, topology, **kwargs):
    if extension not in _TOPOLOGY_EXTS:
        traj = f.read_as_traj(topology, n_frames=n_frames, stride=1, atom_indices=atom_indices, **kwargs)
    else:
        traj = f.read_as_traj(n_frames=n_frames, stride=1, atom_indices=atom_indices, **kwargs)
    return traj


def _efficient_traj_join(trajs):
    assert trajs

    n_frames = sum(t.n_frames for t in trajs)
    concat_traj = preallocate_empty_trajectory(trajs[0].top, n_frames)

    start = 0
    for traj in trajs:
        concat_traj = copy_traj_attributes(concat_traj, traj, start)
        start += traj.n_frames
    return concat_traj