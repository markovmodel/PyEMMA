
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
from copy import copy
from pathlib import Path

from numpy import vstack
import mdtraj as md
import numpy as np
import os



def create_file_reader(input_files, topology, featurizer, chunksize=None, **kw):
    r"""
    Creates a (possibly featured) file reader by a number of input files and either a topology file or a featurizer.
    Parameters
    ----------
    :param input_files:
        A single input file or a list of input files.
    :param topology:
        A topology file. If given, the featurizer argument can be None.
    :param featurizer:
        A featurizer. If given, the topology file can be None.
    :param chunksize:
        The chunk size with which the corresponding reader gets initialized.
    :return: Returns the reader.
    """
    from pyemma.coordinates.data.numpy_filereader import NumPyFileReader
    from pyemma.coordinates.data.py_csv_reader import PyCSVReader
    from pyemma.coordinates.data import FeatureReader
    from pyemma.coordinates.data.fragmented_trajectory_reader import FragmentedTrajectoryReader

    # fragmented trajectories
    if (isinstance(input_files, (list, tuple)) and len(input_files) > 0 and
            any(isinstance(item, (list, tuple)) for item in input_files)):
        return FragmentedTrajectoryReader(input_files, topology, chunksize, featurizer)

    # normal trajectories
    if (isinstance(input_files, str)
            or (isinstance(input_files, (list, tuple))
                and (any(isinstance(item, str) for item in input_files)
                     or len(input_files) == 0))):
        reader = None
        # check: if single string create a one-element list
        if isinstance(input_files, str):
            input_list = [input_files]
        elif len(input_files) > 0 and all(isinstance(item, str) for item in input_files):
            input_list = input_files
        else:
            if len(input_files) == 0:
                raise ValueError("The passed input list should not be empty.")
            else:
                raise ValueError("The passed list did not exclusively contain strings or was a list of lists "
                                 "(fragmented trajectory).")

        # TODO: this does not handle suffixes like .xyz.gz (rare)
        _, suffix = os.path.splitext(input_list[0])

        suffix = str(suffix)

        # check: do all files have the same file type? If not: raise ValueError.
        if all(item.endswith(suffix) for item in input_list):

            # do all the files exist? If not: Raise value error
            all_exist = True
            from six import StringIO
            err_msg = StringIO()
            for item in input_list:
                if not os.path.isfile(item):
                    err_msg.write('\n' if err_msg.tell() > 0 else "")
                    err_msg.write('File %s did not exist or was no file' % item)
                    all_exist = False
            if not all_exist:
                raise ValueError('Some of the given input files were directories'
                                 ' or did not exist:\n%s' % err_msg.getvalue())
            featurizer_or_top_provided = featurizer is not None or topology is not None
            # we need to check for h5 first, because of mdtraj custom HDF5 traj format (which is deprecated).
            if suffix in ('.h5', '.hdf5') and not featurizer_or_top_provided:
                # This check is potentially expensive for lots of files, we also re-open the file twice (causing atime updates etc.)
                # So we simply require that no featurizer option is given.
                # and not all((_is_mdtraj_hdf5_file(f) for f in input_files)):
                from pyemma.coordinates.data.h5_reader import H5Reader
                reader = H5Reader(filenames=input_files, chunk_size=chunksize, **kw)
            # CASE 1.1: file types are MD files
            elif FeatureReader.supports_format(suffix):
                # check: do we either have a featurizer or a topology file name? If not: raise ValueError.
                # create a MD reader with file names and topology
                if not featurizer_or_top_provided:
                    raise ValueError('The input files were MD files which makes it mandatory to have either a '
                                     'Featurizer or a topology file.')

                if suffix in ('.pdb', '.pdb.gz'):
                    raise ValueError('PyEMMA can not read PDB-fake-trajectories. '
                                     'Please consider using a sane trajectory format (e.g. xtc, dcd).')

                reader = FeatureReader(input_list, featurizer=featurizer, topologyfile=topology,
                                       chunksize=chunksize)
            elif suffix in ('.npy', '.npz'):
                reader = NumPyFileReader(input_list, chunksize=chunksize)
            # otherwise we assume that given files are ascii tabulated data
            else:
                reader = PyCSVReader(input_list, chunksize=chunksize, **kw)
        else:
            raise ValueError('Not all elements in the input list were of the type %s!' % suffix)
    else:
        raise ValueError('Input "{}" was no string or list of strings.'.format(input_files))
    return reader


def single_traj_from_n_files(file_list, top):
    """ Creates a single trajectory object from a list of files

    """
    traj = None
    for ff in file_list:
        if isinstance(ff, Path):
            ff = str(ff.resolve())
        if traj is None:
            traj = md.load(copy(ff), top=top)
        else:
            traj = traj.join(md.load(copy(ff), top=top))

    return traj


def copy_traj_attributes(target, origin, start):
    """ Inserts certain attributes of origin into target
    :param target: target trajectory object
    :param origin: origin trajectory object
    :param start: :py:obj:`origin` attributes will be inserted in :py:obj:`target` starting at this index
    :return: target: the md trajectory with the attributes of :py:obj:`origin` inserted
    """

    # The list of copied attributes can be extended here with time
    # Or perhaps ask the mdtraj guys to implement something similar?
    stop = start+origin.n_frames

    target.xyz[start:stop] = origin.xyz
    target.unitcell_lengths[start:stop] = origin.unitcell_lengths
    target.unitcell_angles[start:stop] = origin.unitcell_angles
    target.time[start:stop] = origin.time

    return target


def preallocate_empty_trajectory(top, n_frames=1):
    """

    :param top: md.Topology object to be mimicked in shape
    :param n_frames: desired number of frames of the empty trajectory
    :return: empty_traj: empty md.Trajectory object with n_frames
    """
    # to assign via [] operator to Trajectory objects
    from pyemma.coordinates.util.patches import trajectory_set_item
    md.Trajectory.__setitem__ = trajectory_set_item

    return md.Trajectory(np.zeros((n_frames, top.n_atoms, 3)),
                         top,
                         time=np.zeros(n_frames),
                         unitcell_lengths=np.zeros((n_frames, 3)),
                         unitcell_angles=np.zeros((n_frames, 3))
                         )


def enforce_top(top):
    if isinstance(top, str):
        top = md.load(top).top
    elif isinstance(top, md.Trajectory):
        top = top.top
    elif isinstance(top, md.Topology):
        pass
    else:
        raise TypeError('element %s of the reference list is not of type'
                        'str, md.Trajectory, or md.Topology, but %s' % (top, type(top)))
    return top


def save_traj_w_md_load_frame(reader, sets):
    # Creates a single trajectory object from a "sets" array via md.load_frames
    traj = None
    for file_idx, frame_idx in vstack(sets):
        if traj is None:
            traj = md.load_frame(reader.filenames[file_idx], frame_idx, reader.topfile)
        else:
            traj = traj.join(md.load_frame(reader.filenames[file_idx], frame_idx, reader.topfile))
    return traj


def compare_coords_md_trajectory_objects(traj1, traj2, atom=None, eps=1e-6, mess=False):
    # Compares the coordinates of "atom" for all frames in traj1 and traj2
    # Returns a boolean found_diff and an errmsg informing where
    assert isinstance(traj1, md.Trajectory)
    assert isinstance(traj2, md.Trajectory)
    assert traj1.n_frames == traj2.n_frames, "%i != %i" % (traj1.n_frames, traj2.n_frames)
    assert traj2.n_atoms == traj2.n_atoms

    R = np.zeros((2, traj1.n_frames, 3))
    if atom is None:
        atom_index = np.random.randint(0, high = traj1.n_atoms)
    else:
        atom_index = atom

    # Artificially mess the the coordinates
    if mess:
        traj1.xyz [0, atom_index, 2] += 10*eps

    for ii, traj in enumerate([traj1, traj2]):
        R[ii, :] = traj.xyz[:, atom_index]

    # Compare the R-trajectories among themselves
    found_diff = False
    first_diff = None
    errmsg = ''

    for ii, iR in enumerate(R):
    # Norm of the difference vector
        norm_diff = np.sqrt(((iR - R) ** 2).sum(2))

        # Any differences?
        if (norm_diff > eps).any():
            first_diff = np.argwhere(norm_diff > eps)[0]
            found_diff = True
            errmsg = "Delta R_%u at frame %u: [%2.1e, %2.1e]" % (atom_index, first_diff[1],
                                                                 norm_diff[0, first_diff[1]],
                                                                 norm_diff[1, first_diff[1]])
            errmsg2 = "\nThe position of atom %u differs by > %2.1e for the same frame between trajectories" % (
                atom_index, eps)
            errmsg += errmsg2
            break

    return found_diff, errmsg


def _is_mdtraj_hdf5_file(fh):
    # fh is a h5py.File
    return ('coordinates' in fh.keys() and
            fh.attrs['conventions'] == 'Pande' and
            fh.attrs['conventionVersion'] == '1.1' and
            fh.attrs['program'] == 'MDTraj'
    )
