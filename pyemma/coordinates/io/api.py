r"""

=========================
Pyemma coordinates.io API
=========================

"""

__docformat__ = "restructuredtext en"

"""python package imports"""
import numpy as np

"""emma intra package imports"""
from datareader import DataReader
from datawriter import DataWriter

"""pystallone imports"""
from pyemma.util.pystallone import API as sapi

__author__ = "Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__="m.scherer AT fu-berlin DOT de"

__all__ = ['reader',
           'read_traj',
           'writer',
           'write_traj']

################################################################################
# Molecular IO
################################################################################

# DONE: Map to stallone (Frank)
def reader(filename):
    r"""Opens a trajectory reader to a given trajectory file

    Supports xtc and dcd. For these file types, trajectory frames are presented
    as Nx3 arrays.
    Supports ascii where coordinates are separated by white spaces. For this file
    type, trajectory frames are presented as one-dimensional arrays

    Parameters
    ----------
    filename : str
        The name of the trajectory file

    Returns
    -------
    DataReader :
        Object with access to the specified file

    """
    return DataReader(sapi.dataNew.reader(filename))


# TODO: Map to moltools (Frank)
#def read_molcrd(filename):
#    """
#    Loads a single molecular structure from a structure file
#    
#    Support pdb, gro and charmm crd (ascii and binary)
#    """

# DONE: Map to stallone (Frank)
def read_traj(filename, select = None, frames = None):
    r"""Load the given trajectory completely into memory

    Supports xtc and dcd. For these file types, trajectory frames are presented
    as Nx3 arrays. 
    Supports ascii where coordinates are separated by white spaces. For this file
    type, trajectory frames are presented as one-dimensional arrays

    Parameters
    ----------
    filename : str
        The trajectory filename

    Returns
    -------
    traj : array
        array of size (L,N,3) for a trajectory with L frames and N atoms

    """
    reader = DataReader(sapi.dataNew.reader(filename))
    return reader.load(select=select, frames=frames)


# DONE: Map to stallone (Frank)
def writer(filename, nframes=None, natoms=None):
    r"""Opens a trajectory writer to a given trajectory file

    Supports dcd. For this file type, trajectory frames will be received
    as Nx3 arrays. 
    Supports ascii where coordinates are separated by white spaces. For this file
    type, trajectory frames are presented as one-dimensional arrays

    Parameters
    ----------
    filename : string
        the file name to be written to
    natoms : int
        number of atoms for the writer. Must be given for molecular coordinate
        writers (e.g. dcd) 

    Returns
    -------
    DataWriter : object
       Writer object for MD trajectory

    """
    if (str(filename).lower().endswith('dcd')):
        if (nframes is None) or (natoms is None):
            raise ValueError('To open a dcd writer, please specify nframes and natoms')
        ndim = natoms * 3
        return DataWriter(sapi.dataNew.writer(filename, nframes, ndim))
    else:
        return DataWriter(sapi.dataNew.writerASCII(filename))

# DONE: Map to stallone (Frank)
def write_traj(filename, traj):
    r"""Write complete trajectory to file.

    Supports xtc and dcd. For these file types, trajectory frames will be received
    as (N,3) arrays.

    Parameters
    ----------
    filename : string
        file name
    traj :  ndarray with shape (F,N,3) or (F,d)
        Array containing the full trajectory

    """
    # number of frames to write
    nframes = np.shape(traj)[0]
    # number of dimensions
    ndim = np.shape(traj)[1]
    if (len(np.shape(traj)) > 2):
        ndim *= np.shape(traj)[2]
    # open java writer
    jwriter = sapi.dataNew.writer(filename, nframes, ndim)
    # wrap into python
    writer = DataWriter(jwriter)
    # write trajectory
    writer.addAll(traj)
    # close file
    writer.close()


