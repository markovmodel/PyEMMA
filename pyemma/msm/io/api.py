r"""

=================
PyEMMA MSM io API
=================

"""

__docformat__ = "restructuredtext en"

from pyemma.util.annotators import shortcut

from scipy.sparse import issparse
from scipy.sparse.sputils import isdense

import pyemma.util.discrete_trajectories as trajectory
import matrix.matrix as matrix

__author__ = "Benjamin Trendelkamp-Schroer, Martin Scherer, Frank Noe"
__copyright__ = "Copyright 2014, Computational Molecular Biology Group, FU-Berlin"
__credits__ = ["Benjamin Trendelkamp-Schroer", "Martin Scherer", "Frank Noe"]
__license__ = "FreeBSD"
__version__ = "2.0.0"
__maintainer__ = "Martin Scherer"
__email__="m.scherer AT fu-berlin DOT de"

__all__=['read_discrete_trajectory',
         'read_dtraj',
         'write_discrete_trajectory',
         'write_dtraj',
         'load_discrete_trajectory',
         'load_dtraj',
         'save_discrete_trajectory',
         'save_dtraj',
         'read_matrix',
         'write_matrix',
         'save_matrix',
         'load_matrix',
         ]

################################################################################
# Discrete trajectory IO
################################################################################

################################################################################
# ascii
################################################################################

@shortcut('read_dtraj')
def read_discrete_trajectory(filename):
    r"""Read discrete trajectory from ascii file.   
    
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

    See also
    --------
    write_discrete_trajectory

    Notes
    -----
    The discrete trajectory file contains a single column with
    integer entries.

    Examples
    --------

    >>> from tempfile import NamedTemporaryFile    
    >>> from pyemma.msm.io import write_discrete_trajectory, load_discrete_trajectory

    Use temporary file
    
    >>> tmpfile=NamedTemporaryFile()

    Discrete trajectory
    
    >>> dtraj=np.array([0, 1, 0, 0, 1, 1, 0])

    Write to disk (as ascii file)
    
    >>> write_discrete_trajectory(tmpfile.name, dtraj)

    Read from disk

    >>> X=read_discrete_trajectory(tmpfile.name)
    >>> X
    array([0, 1, 0, 0, 1, 1, 0])
    
    """
    return trajectory.read_discrete_trajectory(filename)

@shortcut('write_dtraj')
def write_discrete_trajectory(filename, dtraj):
    r"""Write discrete trajectory to ascii file.   
    
    Parameters
    ---------- 
    filename : str 
        The filename of the discrete state trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.
    dtraj : array-like of int
        Discrete state trajectory

    See also
    --------
    read_discrete_trajectory

    Notes
    -----
    The discrete trajectory is written to a 
    single column ascii file with integer entries.

    Examples
    --------

    >>> from tempfile import NamedTemporaryFile    
    >>> from pyemma.msm.io import write_discrete_trajectory, load_discrete_trajectory

    Use temporary file
    
    >>> tmpfile=NamedTemporaryFile()

    Discrete trajectory
    
    >>> dtraj=np.array([0, 1, 0, 0, 1, 1, 0])

    Write to disk (as ascii file)
    
    >>> write_discrete_trajectory(tmpfile.name, dtraj)

    Read from disk

    >>> X=read_discrete_trajectory(tmpfile.name)
    >>> X
    array([0, 1, 0, 0, 1, 1, 0])    
    
    """
    trajectory.write_discrete_trajectory(filename, dtraj)


################################################################################
# binary
################################################################################

@shortcut('load_dtraj')
def load_discrete_trajectory(filename):
    r"""Read discrete trajectory form binary file.   

    Parameters
    ---------- 
    filename : str 
        The filename of the discrete state trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.
    
    Returns
    -------
    dtraj : (M,) ndarray of int
        Discrete state trajectory

    See also
    --------
    save_discrete_trajectory

    Notes
    -----
    The binary file is a one dimensional numpy array
    of integers stored in numpy .npy format.

    Examples
    --------

    >>> from tempfile import NamedTemporaryFile    
    >>> from pyemma.msm.io import load_discrete_trajectory, save_discrete_trajectory

    Use temporary file
    
    >>> tmpfile=NamedTemporaryFile()

    Discrete trajectory
    
    >>> dtraj=np.array([0, 1, 0, 0, 1, 1, 0])

    Write to disk (as ascii file)
    
    >>> save_discrete_trajectory(tmpfile.name, dtraj)

    Read from disk

    >>> X=load_discrete_trajectory(tmpfile.name)
    >>> X
    array([0, 1, 0, 0, 1, 1, 0])       
    
    """
    return trajectory.load_discrete_trajectory(filename)

@shortcut('save_dtraj')
def save_discrete_trajectory(filename, dtraj):
    r"""Write discrete trajectory to binary file.  

    Parameters
    ---------- 
    filename : str 
        The filename of the discrete state trajectory file. 
        The filename can either contain the full or the 
        relative path to the file.      
    dtraj : array-like of int
        Discrete state trajectory

    See also
    --------
    load_discrete_trajectory

    Notes
    -----
    The discrete trajectory is stored as ndarray of integers 
    in numpy .npy format.

    Examples
    --------

    >>> from tempfile import NamedTemporaryFile    
    >>> from pyemma.msm.io import load_discrete_trajectory, save_discrete_trajectory

    Use temporary file
    
    >>> tmpfile=NamedTemporaryFile()

    Discrete trajectory
    
    >>> dtraj=np.array([0, 1, 0, 0, 1, 1, 0])

    Write to disk (as ascii file)
    
    >>> save_discrete_trajectory(tmpfile.name, dtraj)

    Read from disk

    >>> X=load_discrete_trajectory(tmpfile.name)
    >>> X
    array([0, 1, 0, 0, 1, 1, 0])       
    
    """
    trajectory.save_discrete_trajectory(filename, dtraj)

################################################################################
# Matrix IO
################################################################################

################################################################################
# ascii
################################################################################

def read_matrix(filename, mode='default', dtype=float, comments='#'):
    r"""Read matrix from ascii file.  
    
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

    See also
    --------
    write_matrix
    
    Notes 
    ----- 
    (M, N) dense matrices are read from ascii files with M rows and N
    columns. Sparse matrices are read from ascii files in coordinate
    list (COO) format and converted to sparse matrices in (COO)
    format.

    The dtype and comments options do only apply to
    reading and writing of ascii files.

    Examples
    --------

    >>> from tempfile import NamedTemporaryFile    
    >>> from pyemma.msm.io import read_matrix, write_matrix

    **dense**

    Use temporary file with ending '.dat'

    >>> tmpfile=NamedTemporaryFile(suffix='.dat')

    Dense (3, 2) matrix

    >>> A=np.array([[3, 1], [2, 1], [1, 1]])
    >>> write_matrix(tmpfile.name, A)

    Load from disk

    >>> X=load_matrix(tmpfile.name)
    >>> X
    array([[ 3.,  1.],
           [ 2.,  1.],
           [ 1.,  1.]])
           
    **sparse**
    
    >>> from scipy.sparse import csr_matrix

    Use temporary file with ending '.coo.dat'

    >>> tmpfile=NamedTemporaryFile(suffix='.coo.dat')

    Sparse (3, 3) matrix

    >>> A=csr_matrix(np.eye(3))
    >>> write_matrix(tmpfile.name, A)

    Load from disk

    >>> X=load_matrix(tmpfile.name)
    >>> X
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
           
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

def write_matrix(filename, A, mode='default', fmt='%.18e', header='', comments='#'):
    r"""Write matrix to ascii file.  
    
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

    See also
    --------
    read_matrix
    
    Notes
    -----
    (M, N) dense matrices are stored as ascii file with M rows and N
    columns. Sparse matrices are converted to coordinate list (COO)
    format. The coordinate list [...,(row, col, value),...]  is then
    stored as a dense (K, 3) ndarray. K is the number of nonzero
    entries in the sparse matrix.

    Using the naming scheme name.xxx for dense matrices and
    name.coo.xxx for sparse matrices will allow read_matrix to
    automatically infer the appropriate matrix type from the given
    filename.

    Examples
    --------

    >>> from tempfile import NamedTemporaryFile    
    >>> from pyemma.msm.io import read_matrix, write_matrix

    **dense**

    Use temporary file with ending '.dat'

    >>> tmpfile=NamedTemporaryFile(suffix='.dat')

    Dense (3, 2) matrix

    >>> A=np.array([[3, 1], [2, 1], [1, 1]])
    >>> write_matrix(tmpfile.name, A)

    Load from disk

    >>> X=load_matrix(tmpfile.name)
    >>> X
    array([[ 3.,  1.],
           [ 2.,  1.],
           [ 1.,  1.]])
           
    **sparse**
    
    >>> from scipy.sparse import csr_matrix

    Use temporary file with ending '.coo.dat'

    >>> tmpfile=NamedTemporaryFile(suffix='.coo.dat')

    Sparse (3, 3) matrix

    >>> A=csr_matrix(np.eye(3))
    >>> write_matrix(tmpfile.name, A)

    Load from disk

    >>> X=load_matrix(tmpfile.name)
    >>> X
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    
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

def save_matrix(filename, A, mode='default'):
    r"""Save matrix as binary file.  
    
    Parameters
    ----------
    filename : str
        Relative or absolute pathname of the output file.
    A : (M, N) ndarray or sparse matrix
    mode : {'default', 'dense', 'sparse'}
    
        ========== ===================================================
         mode
        ========== ===================================================
        'default'   Use the type of A to determine the format\
                    name.xxx (dense), name.coo.xxx (sparse)      
        'dense'     Enforce conversion to a dense representation\
                    and store the corresponding ndarray
        'sparse'    Convert to sparse matrix in COO-format\
                    and store the coordinate list as ndarray
        ========== ===================================================

    See also
    --------
    load_matrix
    
    Notes
    -----
    (M, N) dense matrices are stored as ndarrays in numpy .npy binary
    format. Sparse matrices are converted to coordinate list (COO)
    format. The coordinate list [...,(row, col, value),...]  is then
    stored as a (K, 3) ndarray in numpy .npy binary format.  K is the
    number of nonzero entries in the sparse matrix.

    Using the naming scheme name.npy for dense matrices 
    and name.coo.npy for sparse matrices will allow
    load_matrix to automatically infer the appropriate matrix
    type from the given filename.

    Examples
    --------

    >>> from tempfile import NamedTemporaryFile    
    >>> from pyemma.msm.io import load_matrix, save_matrix

    **dense**

    Use temporary file with ending '.npy'

    >>> tmpfile=NamedTemporaryFile(suffix='.npy')

    Dense (3, 2) matrix

    >>> A=np.array([[3, 1], [2, 1], [1, 1]])
    >>> write_matrix(tmpfile.name, A)

    Load from disk

    >>> X=load_matrix(tmpfile.name)
    >>> X
    array([[ 3.,  1.],
           [ 2.,  1.],
           [ 1.,  1.]])
           
    **sparse**
    
    >>> from scipy.sparse import csr_matrix

    Use temporary file with ending '.coo.dat'

    >>> tmpfile=NamedTemporaryFile(suffix='.coo.npy')

    Sparse (3, 3) matrix

    >>> A=csr_matrix(np.eye(3))
    >>> write_matrix(tmpfile.name, A)

    Load from disk

    >>> X=load_matrix(tmpfile.name)
    >>> X
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
    
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
    r"""Read matrix from binary file.  
    
    Parameters
    ----------
    filename : str
        Relative or absolute pathname of the input file.
    mode : {'default', 'dense', 'sparse'}

        ========== ====================================================
         mode
        ========== ====================================================
        'default'   Use the filename to determine the matrix format\
                    name.npy (dense), name.coo.npy (sparse)       
        'dense'     Read file as dense matrix 
        'sparse'    Read file as sparse matrix in COO-format
        ========== ====================================================

    See also
    --------
    save_matrix

    Notes
    -----
    (M, N) dense matrices are read as ndarray from binary numpy .npy
    files. Sparse matrices are read as ndarray representing a
    coordinate list [...,(row, col, value),...]  from binary numpy
    .npy files and returned as sparse matrices in (COO) format.

    Examples
    --------

    >>> from tempfile import NamedTemporaryFile    
    >>> from pyemma.msm.io import load_matrix, save_matrix

    **dense**

    Use temporary file with ending '.npy'

    >>> tmpfile=NamedTemporaryFile(suffix='.npy')

    Dense (3, 2) matrix

    >>> A=np.array([[3, 1], [2, 1], [1, 1]])
    >>> write_matrix(tmpfile.name, A)

    Load from disk

    >>> X=load_matrix(tmpfile.name)
    >>> X
    array([[ 3.,  1.],
           [ 2.,  1.],
           [ 1.,  1.]])
           
    **sparse**
    
    >>> from scipy.sparse import csr_matrix
    
    Use temporary file with ending '.coo.dat'
    
    >>> tmpfile=NamedTemporaryFile(suffix='.coo.npy')
    
    Sparse (3, 3) matrix
    
    >>> A=csr_matrix(np.eye(3))
    >>> write_matrix(tmpfile.name, A)
    
    Load from disk
    
    >>> X=load_matrix(tmpfile.name)
    >>> X
    array([[ 1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.]])
        
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

    
