###############################################################################################################################################
#
# Count matrix tools
#
###############################################################################################################################################

import emma2.msm.util
import numpy as np

def is_connected(disc):
    """
        returns True if all states are connected, or False otherwise.
        disc: list of files or list of integer lists or square matrix
    """

def connected_sets(disc):
    """
        returns the largest connected set
        disc: list of files or list of integer lists or square matrix
    """
    
def largest_connected_set(disc):
    """
        returns the largest connected set
        disc: list of files or list of integer lists or square matrix
    """

def countmatrix(dtrajs, lag=1, subset=None):
    """
        count matrix on the given set of states
        dtrajs: list of files or list of integer lists corresponding to discrete trajectories
        lag: lag time in time units
    """

def countmatrix_cores(dtrajs, cores, lag=1):
    """
        milestoning count matrix for given set of cores
        dtrajs: list of files or list of integer lists corresponding to discrete trajectories
        cores: list of integer sets (the cores)
        lag: lag time in time units
    """

###############################################################################################################################################
#
# Transition matrix estimation tools
#
###############################################################################################################################################

def estimate_T(C, reversible=True, statdist=None):
    """
        estimates the transition matrix
    """

def statdist(T):
    """
        compute the stationary distribution of T
    """
    
#def its([dtraj], [lags], reversible=true)


###############################################################################################################################################
#
# Transition matrix assessment tools
#
###############################################################################################################################################

def is_transitionmatrix(T):
    """
        True if T is a transition matrix
    """
    return emma2.msm.util.isProbabilisticMatrix(T)

def is_rate_matrix(K):
    """
    Checks nonnegativity of a matrix.

    Matrix A=(a_ij) is nonnegative if 

        a_ij>=0 for all i, j.

    Parameters :
    ------------
    A : ndarray, shape=(M, N)
        The matrix to test.

    Returns :
    ---------
    nonnegative : bool
        The truth value of the nonnegativity test.

    Notes :
    -------
    The nonnegativity test is performed using
    boolean ndarrays.

    Nonnegativity is import for transition matrix estimation.

    Examples :
    ----------
    >>> import numpy as np
    >>> A=np.array([[0.4, 0.1, 0.4], [0.2, 0.6, 0.2], [0.3, 0.3, 0.4]])
    >>> x=check_nonnegativity(A)
    >>> x
    True
    
    >>> B=np.array([[1.0, 0.0], [2.0, 3.0]])
    >>> x=check_nonnegativity(A)
    >>> x
    False
    
    
        True if K is a rate matrix
        // check elements
        for (IDoubleIterator it = K.nonzeroIterator(); it.hasNext(); it.advance())
        {
            int i = it.row();
            int j = it.column();
            double kij = it.get();
            if (i == j && kij > 0)
                return(false);
            if (i != j && kij < 0)
                return(false);
        } 
        
        // check row sums
        for (int i=0; i<K.rows(); i++)
        {
            if (Math.abs(Doubles.util.sum(K.viewRow(i))) > 1e-6)
                return(false);
        }
        return(true);
    """
    if not isinstance(K, np.ndarray):
        raise NotImplemented("only impled for NumPy ndarray type")
    
    diag = np.diag(K)
    diag.any()
    
    # check elements
    for i in xrange(0, K.shape[0]):
        for j in xrange(0, K.shape[1]):
            print K[i][j]
            if i == j and K[i][j] > 0:
                return False
            if i != j and K[i][j] < 0:
                return False
    # check row sums
    for i in xrange(0, K.shape[1]):
        if abs(K[i].sum()) > 1e-6:
            return False
        
    return True
        

def is_ergodic(T):
    """
        True if T is connected (irreducible) and aperiodic
    """
    
def log_likelihood(C, T):
    """
        likelihood of C given T
    """
    
def is_reversible(T, statdist=None):
    """
        True if T is a transition matrix
        statdist: tests with respect to this stationary distribution
    """

# ckTest(T, [dtraj], sets)

###############################################################################################################################################
#
# Error estimation
#
###############################################################################################################################################
    
def error_perturbation(C, sensitivity):
    """
        C: count matrix
        sensitivity: sensitivity matrix or tensor of size (m x n x n) where m is the dimension of the target quantity and (n x n) is the size of the transition matrix.
        The sensitivity matrix should be evaluated at an appropriate maximum likelihood or mean of the transition matrix estimated from C.
        returns: (m x m) covariance matrix of the target quantity 
    """
    
def sample_T(C, reversible=True, statdist=None):
    """
        C: count matrix
        reversible: True for detailed balance constraints, false otherwise
        statdist: stationary distribution vector if it should be fixed
    """
    
