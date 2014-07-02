r"""This module implements IO function for discrete state trajectory files

Discrete trajectories are assumed to be stored either as single column
ascii files or as ndarrays of shape (n,) in binary .npy format.

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import numpy as np
import scipy.sparse

################################################################################
# ascii
################################################################################

def read_discrete_trajectory(filename):
    """Read discrete trajectory from ascii file. 

    The ascii file containing a single column with integer entries is
    read into an array of integers.

    Parameters
    ---------- 
    filename : str 
        The filename of the discrete state trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.
    
    Returns
    -------
    dtraj : (M, ) ndarray
        Discrete state trajectory.
    
    """
    with open(filename, "r") as f:
        lines=f.read()
        dtraj=np.fromstring(lines, dtype=int, sep="\n")
        return dtraj    
    
def write_discrete_trajectory(filename, dtraj):
    r"""Write discrete trajectory to ascii file.
    
    The discrete trajectory is written to a 
    single column ascii file with integer entries
    
    Parameters
    ---------- 
    filename : str 
        The filename of the discrete state trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.

    dtraj : array-like
        Discrete state trajectory.
    
    """    
    dtraj=np.asarray(dtraj)
    with open(filename, 'w') as f:
        dtraj.tofile(f, sep='\n', format='%d')

################################################################################
# binary
################################################################################

def load_discrete_trajectory(filename):
    r"""Read discrete trajectory form binary file.

    The binary file is a one dimensional numpy array
    of integers stored in numpy .npy format.

    Parameters
    ---------- 
    filename : str 
        The filename of the discrete state trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.
    
    Returns
    -------
    dtraj : (M,) ndarray
        Discrete state trajectory.
    
    """        
    dtraj=np.load(filename)
    return dtraj

def save_discrete_trajectory(filename, dtraj):
    r"""Write discrete trajectory to binary file.

    The discrete trajectory is stored as ndarray of integers 
    in numpy .npy format.

    Parameters
    ---------- 
    filename : str 
        The filename of the discrete state trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.
    
     
    dtraj : array-like
        Discrete state trajectory.

    """
    dtraj=np.asarray(dtraj)
    np.save(filename, dtraj)
