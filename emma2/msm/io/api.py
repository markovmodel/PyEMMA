r"""

================
Emma2 MSM io API
================

"""

__docformat__ = "restructuredtext en"

from scipy.sparse import issparse
from scipy.sparse.sputils import isdense

import trajectory.trajectory as trajectory
import matrix.matrix as matrix

__all__=['read_discrete_trajectory',
         'write_discrete_trajectory',
         'load_discrete_trajectory', 
         'save_discrete_trajectory',
         'read_matrix',
         'write_matrix', 
         'load_matrix', 
         'save_matrix']

################################################################################
# Discrete trajectory IO
################################################################################

################################################################################
# ascii
################################################################################

# DONE: Implement in Python directly
def read_discrete_trajectory(filename):
    """Read discrete trajectory from ascii file. 

    The discrete trajectory file containing a single column with
    integer entries is read into an array of integers.

    Parameters
    ---------- 
    filename : str
        The filename of the discretized trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.
    
    Returns
    -------
    dtraj : (M, ) ndarray of int
        Discrete state trajectory.
    
    """
    return trajectory.read_discrete_trajectory(filename)

read_dtraj=read_discrete_trajectory

__all__.append('read_dtraj')

# DONE: Implement in Python directly
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

    dtraj : array-like of int
        Discrete state trajectory.
    
    """
    trajectory.write_discrete_trajectory(filename, dtraj)

write_dtraj=shortcut('write_dtraj', write_discrete_trajectory)

################################################################################
# binary
################################################################################

# DONE: Implement in Python directly
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
    dtraj : (M,) ndarray of int
        Discrete state trajectory.
    
    """
    return trajectory.load_discrete_trajectory(filename)

load_dtraj=load_discrete_trajectory

__all__.append('load_dtraj')

# DONE : Implement in Python directly
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
    
     
    dtraj : array-like of int
        Discrete state trajectory.

    """
    trajectory.save_discrete_trajectory(filename, dtraj)

save_dtraj=save_discrete_trajectory

__all__.append('save_dtraj')

################################################################################
# Matrix IO
################################################################################

################################################################################
# ascii
################################################################################

def read_matrix(filename, mode='default', dtype=float, comments='#'):
    r"""Read matrix from ascii file
    
    (M, N) dense matrices are read from ascii files
    with M rows and N columns.
    
    Sparse matrices are read from ascii files in 
    coordinate list (COO) format and converted
    to sparse matrices in (COO) format.
    
    Parameters
    ----------
    filename : str
        Relative or absolute pathname of the input file.
    mode : {'default', 'dense', 'sparse'}       
        How to determine the matrix format
        
        ========== ====================================================
         mode
        ========== ====================================================
        'default'   Use the filename to determine the format
        'dense'     Reads file as dense matrix 
        'sparse'    Reads file as sparse matrix in COO-format
        ========== ====================================================  
    
    dtype : data-type, optional
        Data-type of the resulting array; default is float. 
    comments : str, optional
        The character used to indicate the start of a comment; default: '#'.

    Returns
    -------
    A : (M, N) ndarray or scipy.sparse matrix
        The stored matrix
    
    Notes 
    ----- 
    The dtype and comments options do only apply to
    reading and writing of ascii files.
            
    """    
    if mode=='dense':
        return matrix.read_matrix_dense(filename, dtype=dtype, comments=comments)
    elif mode=='sparse':
        return matrix.read_matrix_sparse(filename, dtype=dtype, comments=comments)
    else:
        is_sparse=matrix.is_sparse_file(filename)
        if is_sparse:
            return matrix.read_matrix_sparse(filename, dtype=dtype, comments=comments)
        else:
            return matrix.read_matrix_dense(filename, dtype=dtype, comments=comments)
    

def write_matrix_ascii(filename, A, mode='default', fmt='%.18e', header='', comments='#'):
    r"""Write matrix to ascii file 
    
    (M, N) dense matrices are stored as ascii file with M rows
    and N columns.
    
    Sparse matrices are converted to coordinate list (COO)
    format. The coordinate list [...,(row, col, value),...]
    is then stored as a dense (K, 3) ndarray. K is the number
    of nonzero entries in the sparse matrix.   
    
    Parameters
    ----------
    filename : str
        Relative or absolute pathname of the output file.
    A : (M, N) ndarray or sparse matrix
    mode : {'default', 'dense', 'sparse'}
        How to determine the storage format
        
        ========== ===============================================
         mode
        ========== ===============================================
        'default'   Use the type of A to determine the format
        'dense'     Enforce conversion to a dense representation\
                    and store the corresponding ndarray
        'sparse'    Convert to sparse matrix in COO-format and\
                    and store the coordinate list as ndarray
        ========== ===============================================

    fmt : str or sequence of strs, optional        
        A single format (%10.5f), a sequence of formats, or a multi-format
        string, e.g. 'Iteration %d - %10.5f', in which case delimiter is
        ignored.
    header : str, optional
        String that will be written at the beginning of the file.
    comments : str, optional
        String that will be prepended to the header strings,
        to mark them as comments. Default: '# '. 
    
    Notes
    -----
    Using the naming scheme name.xxx for dense matrices and
    name.coo.xxx for sparse matrices will allow read_matrix to
    automatically infer the appropriate matrix type from the given
    filename.
    
    """
    if mode=='dense':   
        A=matrix.todense(A)
        matrix.write_matrix_dense(filename, A, fmt=fmt, header=header, comments=comments)
    elif mode=='sparse':
        A=matrix.tosparse(A)
        matrix.write_matrix_sparse(filename, A, fmt=fmt, header=header, comments=comments)
    else:
        if isdense(A):
            matrix.write_matrix_dense(filename, A, fmt=fmt, header=header, comments=comments)
        elif issparse(A):
            matrix.write_matrix_sparse(filename, A, fmt=fmt, header=header, comments=comments)
        else:
            raise TypeError('A is not a numpy.ndarray or a scipy.sparse matrix.')

################################################################################
# binary
################################################################################

def write_matrix_binary(filename, A, mode='default'):
    r"""Save matrix as binary file.
    
    (M, N) dense matrices are stored as ndarrays 
    in numpy .npy binary format
    
    Sparse matrices are converted to coordinate list (COO)
    format. The coordinate list [...,(row, col, value),...]
    is then stored as a (K, 3) ndarray in numpy .npy binary format.
    K is the number of nonzero entries in the sparse matrix.   
    
    Parameters
    ----------
    filename : str
        Relative or absolute pathname of the output file.
    A : (M, N) ndarray or sparse matrix
    mode : {'default', 'dense', 'sparse'}
    
        ========== ===================================================
         mode
        ========== ===================================================
        'default'   Use the type of A to determine the format
        
                    ============== ==================================
                     extension
                    ============== ==================================
                     name.xxx      the file is read as dense matrix
                     name.coo.xxx  the file is read as sparse matrix
                    ============== ==================================
                    
        'dense'     Enforce conversion to a dense representation\
                    and store the corresponding ndarray
        'sparse'    Convert to sparse matrix in COO-format\
                    and store the coordinate list as ndarray
        ========== ===================================================
    
    Notes
    -----
    Using the naming scheme name.npy for dense matrices 
    and name.coo.npy for sparse matrices will allow
    load_matrix to automatically infer the appropriate matrix
    type from the given filename.
    
    """
    if mode=='dense':
        A=matrix.todense(A)
        matrix.save_matrix_dense(filename, A)
    elif mode=='sparse':
        A=matrix.tosparse(A)
        matrix.save_matrix_sparse(filename, A)
    else:
        if isdense(A):
            matrix.save_matrix_dense(filename, A)
        elif issparse(A):
            matrix.save_matrix_sparse(filename, A)
        else:
            raise TypeError('A is not a numpy.ndarray or a scipy.sparse matrix.')      


def load_matrix(filename, mode='default'):
    r"""Read matrix from binary file
    
    (M, N) dense matrices are read as ndarray 
    from binary numpy .npy files.
    
    Sparse matrices are read as ndarray representing
    a coordinate list [...,(row, col, value),...]
    from binary numpy .npy files and returned as
    sparse matrices in (COO) format.
    
    Parameters
    ----------
    filename : str
        Relative or absolute pathname of the input file.
    mode : {'default', 'dense', 'sparse'}

        ========== ====================================================
         mode
        ========== ====================================================
        'default'   Use the filename to determine the matrix format
        
                    ============== ==================================
                     extension        
                    ============== ==================================
                     name.npy       the file is read as dense matrix
                     name.coo.npy   the file is read as sparse matrix
                    ============== ==================================
        
        'dense'     Read file as dense matrix 
        'sparse'    Read file as sparse matrix in COO-format
        ========== ====================================================
    
    """
    if mode=='dense':
        return matrix.load_matrix_dense(filename)
    elif mode=='sparse':
        return matrix.load_matrix_sparse(filename)
    else:
        is_sparse=matrix.is_sparse_file(filename)
        if is_sparse:
            return matrix.load_matrix_sparse(filename)
        else:
            return matrix.load_matrix_dense(filename)

    
