
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on 13.03.2015

@author: marscher
'''
from mdtraj.utils.validation import cast_indices
from mdtraj.core.trajectory import load, _parse_topology, _TOPOLOGY_EXTS, _get_extension, open

from itertools import groupby
from operator import itemgetter


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
    >>>     print chunk

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

    if not isinstance(stride, dict) and chunk % stride != 0:
        pass
        #raise ValueError('Stride must be a divisor of chunk. stride=%d does not go '
        #                 'evenly into chunk=%d' % (stride, chunk))
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

    elif isinstance(stride, list):
        with (lambda x: open(x, n_atoms=topology.n_atoms)
              if extension in ('.crd', '.mdcrd')
              else open(filename))(filename) as f:
            sorted_stride = sorted(stride)
            x_prev = 0
            curr_size = 0
            traj = None
            leftovers = []
            for k, g in groupby(enumerate(sorted_stride), lambda (a, b): a-b):
                grouped_stride = map(itemgetter(1), g)
                f.seek(grouped_stride[0]-x_prev, whence=1)
                x_prev = grouped_stride[-1]
                group_size = len(grouped_stride)
                if curr_size + group_size > chunk:
                    leftovers = grouped_stride[chunk - curr_size:]
                else:
                    local_traj = _get_local_traj_object(atom_indices, extension, f, group_size, topology, **kwargs)
                    if traj:
                        traj.join(local_traj, check_topology=False)
                    else:
                        traj = local_traj

                    curr_size += len(grouped_stride)
                if curr_size == chunk:
                    yield traj
                    curr_size = 0
                    traj = None
                while leftovers:
                    local_chunk = leftovers[:min(chunk, len(leftovers))]
                    local_traj = _get_local_traj_object(atom_indices, extension, f, group_size, topology, **kwargs)
                    if traj:
                        traj.join(local_traj, check_topology=False)
                    else:
                        traj = local_traj
                    leftovers = leftovers[min(chunk, len(leftovers)):]
                    curr_size += len(local_chunk)
                    if curr_size == chunk:
                        yield traj
                        curr_size = 0
                        traj = None

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
