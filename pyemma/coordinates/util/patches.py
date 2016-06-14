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


class iterload(object):

    def __init__(self, filename, chunk=100, **kwargs):
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
        self._filename = filename
        self._stride = kwargs.pop('stride', 1)
        self._atom_indices = cast_indices(kwargs.pop('atom_indices', None))
        self._top = kwargs.pop('top', None)
        self._skip = kwargs.pop('skip', 0)
        self._kwargs = kwargs
        self._chunksize = chunk
        self._extension = _get_extension(self._filename)
        self._closed = False
        if self._extension not in _TOPOLOGY_EXTS:
            self._topology = _parse_topology(self._top)
        else:
            self._topology = self._top

        self._mode = None
        if self._chunksize > 0 and self._extension in ('.pdb', '.pdb.gz'):
            self._mode = 'pdb'
            self._t = load(self._filename, stride=self._stride, atom_indices=self._atom_indices)
            self._i = 0
        elif isinstance(self._stride, np.ndarray):
            self._mode = 'random_access'
            self._f = (lambda x:
                       open(x, n_atoms=self._topology.n_atoms)
                       if self._extension in ('.crd', '.mdcrd')
                       else open(self._filename))(self._filename)
            self._ra_it = self._random_access_generator(self._f)
        else:
            self._mode = 'traj'
            self._f = (
                lambda x: open(x, n_atoms=self._topology.n_atoms)
                if self._extension in ('.crd', '.mdcrd')
                else open(self._filename)
            )(self._filename)

            # offset array handling
            offsets = kwargs.pop('offsets', None)
            if hasattr(self._f, 'offsets') and offsets is not None:
                self._f.offsets = offsets

            if self._skip > 0:
                self._f.seek(self._skip)

    def __iter__(self):
        return self

    def close(self):
        if hasattr(self, '_t'):
            self._t.close()
        elif hasattr(self, '_f'):
            self._f.close()
        self._closed = True

    def __next__(self):
        return self.next()

    def next(self):
        if self._closed:
            raise StopIteration()
        if not isinstance(self._stride, np.ndarray) and self._chunksize == 0:
            # If chunk was 0 then we want to avoid filetype-specific code
            # in case of undefined behavior in various file parsers.
            # TODO: this will first apply stride, then skip!
            if self._extension not in _TOPOLOGY_EXTS:
                self._kwargs['top'] = self._top
            return load(self._filename, stride=self._stride, **self._kwargs)[self._skip:]
        elif self._mode is 'pdb':
            # the PDBTrajectortFile class doesn't follow the standard API. Fixing it
            # to support iterload could be worthwhile, but requires a deep refactor.
            X = self._t[self._i:self._i+self._chunksize]
            self._i += self._chunksize
            return X
        elif isinstance(self._stride, np.ndarray):
            return next(self._ra_it)
        else:
            if self._extension not in _TOPOLOGY_EXTS:
                traj = self._f.read_as_traj(self._topology, n_frames=self._chunksize*self._stride,
                                      stride=self._stride, atom_indices=self._atom_indices, **self._kwargs)
            else:
                traj = self._f.read_as_traj(n_frames=self._chunksize*self._stride,
                                      stride=self._stride, atom_indices=self._atom_indices, **self._kwargs)

            if len(traj) == 0:
                raise StopIteration()

            return traj

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _random_access_generator(self, f):
        with f:
            curr_size = 0
            traj = []
            leftovers = []
            chunksize = self._chunksize
            if chunksize == 0:
                chunksize = np.iinfo(int).max
            for k, g in groupby(enumerate(self._stride), lambda a: a[0] - a[1]):
                grouped_stride = list(map(itemgetter(1), g))
                seek_to = grouped_stride[0] - f.tell()
                f.seek(seek_to, whence=1)
                group_size = len(grouped_stride)
                if curr_size + group_size > chunksize:
                    leftovers = grouped_stride
                else:
                    local_traj = _get_local_traj_object(
                        self._atom_indices, self._extension, f, group_size, self._topology, **self._kwargs)
                    traj.append(local_traj)
                    curr_size += len(grouped_stride)
                if curr_size == chunksize:
                    yield _efficient_traj_join(traj)
                    curr_size = 0
                    traj = []
                while leftovers:
                    local_chunk = leftovers[:min(chunksize, len(leftovers))]
                    local_traj = _get_local_traj_object(
                        self._atom_indices, self._extension, f, len(local_chunk), self._topology, **self._kwargs)
                    traj.append(local_traj)
                    leftovers = leftovers[min(chunksize, len(leftovers)):]
                    curr_size += len(local_chunk)
                    if curr_size == chunksize:
                        yield _efficient_traj_join(traj)
                        curr_size = 0
                        traj = []
            if traj:
                yield _efficient_traj_join(traj)
            raise StopIteration()


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


def trajectory_set_item(self, idx, value):
    """
    :param self: mdtraj.Trajectory
    :param idx: possible slices over frames,
    :param value:
    :return:
    """
    import mdtraj
    assert isinstance(self, mdtraj.Trajectory), type(self)
    if not isinstance(value, mdtraj.Trajectory):
        raise TypeError("value to assign is of incorrect type(%s). Should be mdtraj.Trajectory" % type(value))
    idx = np.index_exp[idx]
    frames, atoms = None, None
    if isinstance(idx, (list, tuple)):
        if len(idx) == 1:
            frames, atoms = idx[0], slice(None, None, None)
        if len(idx) == 2:
            frames, atoms = idx[0], idx[1]
        if len(idx) >= 3 or len(idx) == 0:
            raise IndexError("invalid slice by %s" % idx)

    print("frames: %s\tatoms: %s" %(frames, atoms))
    self.xyz[frames, atoms] = value.xyz
    self._time[frames] = value.time
    self.unitcell_lengths[frames] = value.unitcell_lengths
    self.unitcell_angles[frames] = value.unitcell_angles
