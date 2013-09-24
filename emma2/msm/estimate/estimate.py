###############################################################################################################################################
#
# Count matrix tools
#
###############################################################################################################################################


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

def is_rate_matrix(K):
    """
        True if K is a rate matrix
    """

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
    
