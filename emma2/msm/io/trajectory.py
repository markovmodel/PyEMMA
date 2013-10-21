"""This module implements IO function for discrete state trajectory files

Discrete trajectories are assumed to be stored either as single column
ascii files or as ndarrays of shape (n,) in binary .npy format."""

import numpy as np
import scipy.sparse

################################################################################
# ascii
################################################################################

def read_discrete_trajectory(filename):
    r"""Read one or multiple ascii textfiles,
    each containing a single column with integer
    entries into a list of integers.
    
    Parameters
    ---------- 
    filename : str or list of str
        The filename of the discretized trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.
    
    Returns
    -------
    dtraj : array-like (or list of array-like)
        An array with integer entries.
    
    """
    if isinstance(filename, str):
        dtraj=read_discrete_trajectory_single(filename)
        return dtraj
    else:
        dtraj=[]
        for name in filename:
            dtraj.append(read_discrete_trajectory_single(name))
        return dtraj        
    
def read_discrete_trajectory_single(filename):
    r"""Read one ascii text file

    The file contains a single column with
    integer entries.

    Parameters
    ---------- 
    filename : str or list of str
        The filename of the discretized trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.

    Returns
    -------
    dtraj : array
        An array with integer entries.
        
    """
    f=open(filename, "r")
    lines=f.read()
    dtraj=np.fromstring(lines, dtype=int, sep="\n")
    return dtraj

def write_discrete_trajectory(filename, dtraj):
    r"""Write one or multiple discrete trajectories.
    
    Each discrete trajectory is written to a 
    single column ascii text file.
    
    Parameters
    ----------
    filename : str or list of str
        The filename of the discretized trajectory file. 
        The filename can either contain the full or the 
        relative path to the file. For a list of discrete
        trajectories filename can also be a directory name
        instead of a list of filenames.

    dtraj : array-like (or list of array-like)
        An array with integer entries.
    
    """    
    if isinstance(filename, str):
        write_discrete_trajectory_single(filename, dtraj)
    else:
        for i in range(len(filename)):
            write_discrete_trajectory_single(filename[i], dtraj[i])
        

def write_discrete_trajectory_single(filename, dtraj):
    r"""Write one or multiple discrete trajectories.
    
    Each discrete trajectory is written to a 
    single column ascii text file.
    
    Parameters
    ----------
    filename : str or list of str
        The filename of the discretized trajectory file. 
        The filename can either contain the full or the 
        relative path to the file. For a list of discrete
        trajectories filename can also be a directory name
        instead of a list of filenames.

    dtraj : array-like (or list of array-like)
        An array with integer entries.
    
    """
    with open(filename, 'w') as f:
        dtraj.tofile(f, sep='\n', format='%d')
    

    
