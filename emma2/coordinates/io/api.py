'''
Created on Dec 30, 2013

@author: noe
'''

# emma imports
from datareader import DataReader
from datawriter import DataWriter
# stallone imports
import emma2.util.pystallone as stallone
from emma2.util.pystallone import API as sapi

###################################################################################################
# Molecular IO
###################################################################################################


# DONE: Map to stallone (Frank)
def reader(filename):
    """
    Opens a trajectory reader to a given trajectory file
    
    Supports xtc and dcd. For these file types, trajectory frames are presented
    as Nx3 arrays. 
    Supports ascii where coordinates are separated by white spaces. For this file
    type, trajectory frames are presented as one-dimensional arrays
    
    Returns
    -------
    A DataReader object with access to the specified file
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
    """
    Loads the given trajectory completely into memory
    
    Supports xtc and dcd. For these file types, trajectory frames are presented
    as Nx3 arrays. 
    Supports ascii where coordinates are separated by white spaces. For this file
    type, trajectory frames are presented as one-dimensional arrays
    
    Returns
    -------
    traj : array
        array of size (F,N,3) for a trajectory with F frames and N atoms
    """
    reader = DataReader(sapi.dataNew.reader(filename))
    return reader.load(select=select, frames=frames)


# DONE: Map to stallone (Frank)
def writer(filename, nframes = None, natoms = None):
    """
    Opens a trajectory writer to a given trajectory file
    
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
    """
    if (str(filename).lower().endswith('dcd')):
        if (nframes is None) or (natoms is None):
            raise ValueError('To open a dcd writer, please specify nframes and natoms')
        return DataWriter(sapi.dataNew.writer(filename,nframes,natoms))
    else:
        return DataWriter(sapi.dataNew.writerASCII(filename))

# DONE: Map to stallone (Frank)
def write_traj(filename, traj):
    """
    Opens a trajectory writer to a given trajectory file
    
    Supports xtc and dcd. For these file types, trajectory frames will be received
    as Nx3 arrays. 
    """
    writer = DataWriter(sapi.dataNew.writer(filename))
    reader.addAll(traj)
    reader.close()


