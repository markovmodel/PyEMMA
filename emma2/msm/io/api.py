"""This is the api-definition for the io module"""

################################################################################
# Discrete trajectory IO
################################################################################

# TODO: Implement in Python directly
def read_discrete_trajectory(filename):
    """Read one or multiple ascii textfiles,
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
        A list with integer entries.
    
    """
# TODO: Implement in Python directly
def load_discrete_trajectory(filename):
    r"""Read one or multiple binary files,
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
        A list with integer entries.
    
    """

# TODO: Implement in Python directly
def read_dtraj(filename):
    """
    (short version of the above)
    """
# TODO: Implement in Python directly
def load_dtraj(filename):
    """
    (short version of above)
    """

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
    
    """

def save_discrete_trajectory(filename, dtraj):
    r"""Write one or multiple discrete trajectories
    
    Each discrete trajectory is written as 
    a numpy ndarray to a numpy binary .npy file.
    
    Parameters
    ----------
    filename : str or list of str
        The filename of the discretized trajectory file. 
        The filename can either contain the full or the 
        relative path to the file. For a list of discrete
        trajectories filename can also be a directory name
        instead of a list of filenames.
    
    """
    
def write_dtraj(filename, dtraj):
    r"""Short version
    
    """

def save_dtraj(filename, dtraj):
    r"""Short version

    """

################################################################################
# Matrix IO
################################################################################

def write_matrix(filename, A, fmt='default'):
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
    fmt : {'default', 'dense', 'sparse'}
        'default' : Use the type of A to determine if to store
            in dense or sparse format.
        'dense' : Enforces conversion to a dense representation
            and stores the corresponding ndarray.
        'sparse' : Converts to sparse matrix in COO-format and
            and stores the coordinate list as dense array.
    
    Notes
    -----
    Using the naming scheme name.xxx for dense matrices 
    and name.coo.xxx for sparse matrices will allow
    read_matrix to automatically return the appropriate matrix
    type from the given filename.                       
    
    """

def save_matrix(filename, A, fmt='default'):
    r"""Write matrix to ascii file 
    
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
    fmt : {'default', 'dense', 'sparse'}
        'default' : Use the type of A to determine if to store
            in dense or sparse format.
        'dense' : Enforces conversion to a dense representation
            and stores the corresponding ndarray.
        'sparse' : Converts to sparse matrix in COO-format and
            and stores the coordinate list as dense array.   
    
    Notes
    -----
    Using the naming scheme name.npy for dense matrices 
    and name.coo.npy for sparse matrices will allow
    read_matrix to automatically return the appropriate matrix
    type from the given filename.
    
    """    

def read_matrix(filename, fmt='default'):
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
    fmt : {'default', 'dense', 'sparse'}
        'default' : Use the filename to determine the matrix format.
            name.xxx : the file is read as dense matrix
            name.coo.xxx :  the file is read as sparse matrix
        'dense' : Reads file as dense matrix 
        'sparse' : Reads file as sparse matrix in COO-format.
    
    """

def load_matrix(filename, fmt='default'):
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
    fmt : {'default', 'dense', 'sparse'}
        'default' : Use the filename to determine the matrix format.
            name.npy : the file is read as dense matrix
            name.coo.npy :  the file is read as sparse matrix
        'dense' : Reads file as dense matrix 
        'sparse' : Reads file as sparse matrix in COO-format.
    
    """





    
