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

from collections import namedtuple

import numpy as np
from mdtraj import FormatRegistry
from mdtraj import Topology, Trajectory
from mdtraj.utils import in_units_of
from mdtraj.utils.validation import cast_indices
from mdtraj.core.trajectory import load, _TOPOLOGY_EXTS, _get_extension, open as md_open, load_topology

from itertools import groupby
from operator import itemgetter, attrgetter

from pyemma.coordinates.data.util.reader_utils import copy_traj_attributes, preallocate_empty_trajectory
from six.moves import map

TrajData = namedtuple("traj_data", ('xyz', 'unitcell_lengths', 'unitcell_angles', 'box'))


def _cache_mdtraj_topology(args):
    import hashlib
    from mdtraj import load_topology as md_load_topology
    _top_cache = {}

    def wrap(top_file):
        if isinstance(top_file, Topology):
            return top_file
        if isinstance(top_file, Trajectory):
            return top_file.topology
        hasher = hashlib.md5()
        with open(top_file, 'rb') as f:
            hasher.update(f.read())
        hash = hasher.hexdigest()

        if hash in _top_cache:
            top = _top_cache[hash]
        else:
            top = md_load_topology(top_file)
            _top_cache[hash] = top
        return top

    return wrap


load_topology_cached = _cache_mdtraj_topology(load_topology)


class iterload(object):
    def __init__(self, filename, chunk=1000, **kwargs):
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
        self._seeked = False
        if self._extension not in _TOPOLOGY_EXTS:
            self._topology = load_topology_cached(self._top)
        else:
            self._topology = self._top

        if self._extension in ('pdb', 'pdb.gz'):
            raise Exception("Not supported as trajectory format {ext}".format(ext=self._extension))

        self._mode = None
        if isinstance(self._stride, np.ndarray):
            self._mode = 'random_access'
            self._f = (lambda x:
                       md_open(x, n_atoms=self._topology.n_atoms)
                       if self._extension in ('.crd', '.mdcrd')
                       else md_open(self._filename))(self._filename)
            self._ra_it = self._random_access_generator(self._f)
        else:
            self._mode = 'traj'
            self._f = (
                lambda x: md_open(x, n_atoms=self._topology.n_atoms)
                if self._extension in ('.crd', '.mdcrd')
                else md_open(self._filename)
            )(self._filename)

            # offset array handling
            offsets = kwargs.pop('offsets', None)
            if hasattr(self._f, 'offsets') and offsets is not None:
                self._f.offsets = offsets

    @property
    def skip(self):
        return self._skip

    @skip.setter
    def skip(self, value):
        assert self._mode == 'traj'
        self._skip = value

    def __iter__(self):
        return self

    def close(self):
        if hasattr(self, '_f'):
            self._f.close()
        self._closed = True

    def __next__(self):
        return self.next()

    def next(self):
        if self._closed:
            raise StopIteration("closed file")

        # apply skip offset only once.
        # (we want to do this here, since we want to be able to re-set self.skip)
        if self.skip > 0 and not self._seeked:
            try:
                self._f.seek(self.skip)
                self._seeked = True
            except (IOError, IndexError):
                raise StopIteration("too short trajectory")

        if not isinstance(self._stride, np.ndarray) and self._chunksize == 0:
            # If chunk was 0 then we want to avoid filetype-specific code
            # in case of undefined behavior in various file parsers.
            # TODO: this will first apply stride, then skip!
            if self._extension not in _TOPOLOGY_EXTS:
                self._kwargs['top'] = self._top
            traj = load(self._filename, stride=self._stride, **self._kwargs)[self.skip:]
        elif isinstance(self._stride, np.ndarray):
            return next(self._ra_it)
        else:
            if self._extension not in _TOPOLOGY_EXTS:
                traj = self._f.read_as_traj(self._topology, n_frames=self._chunksize * self._stride,
                                            stride=self._stride, atom_indices=self._atom_indices, **self._kwargs)
            else:
                traj = self._f.read_as_traj(n_frames=self._chunksize * self._stride,
                                            stride=self._stride, atom_indices=self._atom_indices, **self._kwargs)

        if len(traj) == 0:
            raise StopIteration("eof")

        return traj

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _random_access_generator(self, f):
        with f:
            curr_size = 0
            coords = []
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
                    local_traj_data = _read_traj_data(self._atom_indices, f, group_size, **self._kwargs)
                    coords.append(local_traj_data)
                    curr_size += len(grouped_stride)
                if curr_size == chunksize:
                    yield _join_traj_data(coords, self._topology)
                    curr_size = 0
                    coords = []
                while leftovers:
                    local_chunk = leftovers[:min(chunksize, len(leftovers))]
                    local_traj_data = _read_traj_data(self._atom_indices, f, len(local_chunk), **self._kwargs)
                    coords.append(local_traj_data)
                    leftovers = leftovers[min(chunksize, len(leftovers)):]
                    curr_size += len(local_chunk)
                    if curr_size == chunksize:
                        yield _join_traj_data(coords, self._topology)
                        curr_size = 0
                        coords = []
            if coords:
                yield _join_traj_data(coords, self._topology)

            raise StopIteration("delivered all RA indices")


def _read_traj_data(atom_indices, f, n_frames, **kwargs):
    """
    
    Parameters
    ----------
    atom_indices
    f : file handle of mdtraj
    n_frames
    kwargs

    Returns
    -------
    data : TrajData(xyz, unitcell_length, unitcell_angles, box)
    
    Format read() return values:
     amber_netcdf_restart_f: xyz [Ang], time, cell_l, cell_a
     amber restart: xyz[Ang], time, cell_l, cell_a

     hdf5: xyz[nm], time, cell_l, cell_a
     dtr: xyz[Ang], time, cell_l, cell_a
     netcdf: xyz [ang], time, cell_l, cell_a

     lammps: xyz, cell_l, cell_a
     dcd: xyz[Ang], cell_l, cell_a

     mdcrd: xyz [ang], cell_l (can be none?)

     gro: xyz[nm], time, unit_cell_vectors [in nano meters] ....

     trr: xyz[nm], time, step, box (n, 3, 3), lambd?
     xtc: xyz[nm], time, step, box
     
     xyz: xyz
     lh5: xyz [nm]
     arc: xyz[Ang]
     binpos: xyz[Ang]

    """
    res = f.read(n_frames=n_frames, stride=1, atom_indices=atom_indices, **kwargs)
    if isinstance(res, np.ndarray):
        res = [res]

    # first element is always xyz coords array.
    xyz = res[0]

    # hopefully this works for all formats?
    xyz = in_units_of(xyz, f.distance_unit, Trajectory._distance_unit, inplace=True)

    box = cell_lengths = cell_angles = None

    from mdtraj.formats import (XTCTrajectoryFile, TRRTrajectoryFile, GroTrajectoryFile, MDCRDTrajectoryFile,
                                LAMMPSTrajectoryFile, DCDTrajectoryFile, HDF5TrajectoryFile, DTRTrajectoryFile,
                                NetCDFTrajectoryFile)

    if isinstance(f, (XTCTrajectoryFile, TRRTrajectoryFile)):
        box = res[3]
    elif isinstance(f, GroTrajectoryFile):
        box = res[2]
    elif isinstance(f, MDCRDTrajectoryFile):
        cell_lengths = res[1]
        if cell_lengths is None:
            cell_angles = None
        else:
            # Assume that its a rectilinear box
            cell_angles = 90.0 * np.ones_like(cell_lengths)
    elif isinstance(f, (LAMMPSTrajectoryFile, DCDTrajectoryFile)):
        cell_lengths, cell_angles = res[1:]
    elif len(res) == 4 or isinstance(f, (HDF5TrajectoryFile, DTRTrajectoryFile, NetCDFTrajectoryFile)):
        cell_lengths, cell_angles = res[2:4]
    else:
        assert len(res) == 1, "len:{l}, type={t}".format(l=len(res), t=f)
        #raise NotImplementedError("format read function not handled..." + str(f))

    box = in_units_of(box, f.distance_unit, Trajectory._distance_unit, inplace=True)

    return TrajData(xyz, cell_lengths, cell_angles, box)


def _join_traj_data(traj_data, top_file):
    top = load_topology_cached(top_file)
    xyz = np.concatenate(tuple(map(itemgetter(0), traj_data)))

    traj = Trajectory(xyz, top)

    if all(t.unitcell_lengths is not None for t in traj_data):
        unitcell_lengths = np.concatenate(tuple(map(itemgetter(1), traj_data)))
        traj.unitcell_lengths = unitcell_lengths

    if all(t.box is not None for t in traj_data):
        boxes = np.concatenate(tuple(map(itemgetter(-1), traj_data)))
        traj.unitcell_vectors = boxes

    if all(t.unitcell_angles is not None for t in traj_data):
        angles = np.concatenate(tuple(map(itemgetter(2), traj_data)))
        traj.unitcell_angles = angles

    return traj


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

    self.xyz[frames, atoms] = value.xyz
    self._time[frames] = value.time
    self.unitcell_lengths[frames] = value.unitcell_lengths
    self.unitcell_angles[frames] = value.unitcell_angles
