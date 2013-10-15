################################################################################
# Discrete trajectory IO
################################################################################

# TODO: Implement in Python directly
def read_discrete_trajectory(filename):
    """
    Read one or multiple ascii textfiles, each containing a single column with integer
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

################################################################################
# Count matrix
################################################################################

# TODO: Benjamin Implement in Python directly
def count_matrix(dtraj, lag, sliding=True):
    r"""Generate a count matrix in from a given list of integers.

    Parameters
    ----------
    dtraj : array_like
        Discretized trajectory
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.

    Returns
    -------
    C : scipy.sparse.coo_matrix
        The countmatrix at given lag in coordinate list format.
    
    """

# TODO: Benjamin Implement in Python directly
def cmatrix(dtraj, lag, sliding=True):
    r"""Generate a count matrix in from a given list of integers.

    Parameters
    ----------
    dtraj : array_like
        Discretized trajectory
    lag : int
        Lagtime in trajectory steps
    sliding : bool, optional
        If true the sliding window approach 
        is used for transition counting.

    Returns
    -------
    C : scipy.sparse.coo_matrix
        The countmatrix at given lag in coordinate list format.
        
    """

# TODO: Jan Implement in Python directly
def cmatrix_cores(dtraj, cores, lag, sliding=True):
    r"""Generate a countmatrix for the milestoning process on the given core sets
    """

################################################################################
# Connectivity
################################################################################

# TODO: Ben Implement in Python directly
def connected_sets(C):
    """
    Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.

    Returns
    -------
    cc : list of lists of integers
        Each entry is a list containing all vertices (states) in 
        the corresponding connected component.

    """

# TODO: Ben Implement in Python directly
def largest_connected_set(C):
    """
    Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.

    Returns
    -------
    lcc : list of integers
        The largest connected component of the directed graph.

    """

# TODO: Ben Implement in Python directly
def connected_cmatrix(C):
    """
    Compute the count matrix of the largest connected set.

    The input count matrix is used as a weight matrix for the
    construction of a directed graph. The largest connected set of
    the constructed graph is computed. Vertices belonging to the
    largest connected component are used to generate a completely
    connected subgraph. The weight matrix of the subgraph is the
    desired completely connected count matrix.
    
    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.

    Returns
    -------
    C_cc : scipy.sparse matrix
        Count matrix of largest completely 
        connected set of vertices (states)

    """

# TODO: Jan Implement in Python directly
def is_connected(C):
    """Return true if C is a countmatrix for a completely connected process.    
    """

# TODO: Implement in Python directly
def mapping(set):
    """
    Constructs two dictionaries that map from the set values to their indexes, and vice versa.
    
    Parameters
    ----------
    set : array-like of integers 

    Returns
    -------
    dict : python dictionary mapping original to internal states 
    dict : python dictionary mapping internal to original states 

    """
    


################################################################################
# Transition matrix
################################################################################

# TODO: Jan Implement in Python directly (Nonreversible)
# TODO: Implement in Python directly (Reversible with stat dist)
# TODO: Martin Map to Stallone (Reversible)
def transition_matrix(C, reversible=False, mu=None, **kwargs):
    """
    Estimate the transition matrix from the given countmatrix.

    The transition matrix is a maximum likelihood estimate (MLE)
    of the probability distribution of transition matrices
    with parameters given by the countmatrix. 

    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix
    reversible : bool (optional)
        If True restrict the ensemble of transition matrices
        to those having a detailed balance symmetry otherwise
        the likelihood optimization is carried out over the whole
        space of stochastic matrices.
    mu : array_like
        The stationary distribution of the MLE transition matrix.
    **kwargs: Optional algorithm-specific parameters

    Returns
    -------
    P : numpy ndarray, shape=(n, n) or scipy.sparse matrix
       The MLE transition matrix

    """

# TODO: Jan Implement in Python directly (Nonreversible)
# TODO: Implement in Python directly (Reversible with stat dist)
# TODO: Map to Stallone (Reversible)
def tmatrix(C, reversible=False, mu=None):
    r"""Estimate the transition matrix from the given countmatrix.
    """

# TODO: Jan Implement in Python directly
def tmatrix_cov(C, k=None):
    """
    Computes a nonreversible covariance matrix of transition matrix elments
    
    Parameters
    ----------
    C : scipy.sparse matrix
        Count matrix
    k : row index (optional). 
        If set, only the covariance matrix for this row is returned.
        
    """
    
# TODO: Jan Implement in Python directly
def log_likelihood(C, T):
    """
        likelihood of C given T
    """
    
# TODO: Implement in Python directly
def error_perturbation(C, sensitivity):
    """
        C: count matrix
        sensitivity: sensitivity matrix or tensor of size (m x n x n) where m is the dimension of the target quantity and (n x n) is the size of the transition matrix.
        The sensitivity matrix should be evaluated at an appropriate maximum likelihood or mean of the transition matrix estimated from C.
        returns: (m x m) covariance matrix of the target quantity 
    """

# TODO: Martin Map to Stallone (Reversible)
def tmatrix_sampler(C, reversible=False, mu=None, P0=None):
    """
    Parameters
    ----------
    C : ndarray, shape=(n, n) or scipy.sparse matrix
        Count matrix    
    reversible : bool
        If true sample from the ensemble of transition matrices
        restricted to those obeying a detailed balance condition,
        else draw from the whole ensemble of stochatic matrices.
    mu : array_like
        The stationary distribution of the transition matrix samples.
    P0 : ndarray, shape=(n, n) or scipy.sparse matrix
        Starting point of the MC chain of the sampling algorithm.
        Has to obey the required constraints.    

    Returns :
    ---------
    sampler : A TransitionMatrixSampler object.
        
    """



