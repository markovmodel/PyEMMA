"""This module implements the connectivity functionality"""

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph as csgraph

def connected_sets(C):
    r"""Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.

    Returns
    -------
    cc : list of arrays of integers
        Each entry is an array containing all vertices (states) in 
        the corresponding connected component.

    """
    M=C.shape[0]
    """ Compute connected components of C. nc is the number of
    components, indices contain the component labels of the states
    """
    nc, indices=csgraph.connected_components(C, directed=True, connection='strong')
    
    states=np.arange(M) # Discrete states

    """Order indices"""
    ind=np.argsort(indices)
    indices=indices[ind]

    """Order states"""
    states=states[ind]
    """ The state index tuple is now of the following form (states,
    indices)=([s_23, s_17,...,s_3, s_2, ...], [0, 0, ..., 1, 1, ...])
    """

    """Find number of states per component"""
    count=np.bincount(indices)

    """Cumulative sum of count gives start and end indices of
    components"""
    csum=np.zeros(len(count)+1)
    csum[1:]=np.cumsum(count)

    """Generate list containing components, sort each conponent by
    increasing state label"""
    cc=[]
    for i in range(nc):
        cc.append(np.sort(states[csum[i]:csum[i+1]]))
    
    """Sort by size of component - largest component first"""
    cc=sorted(cc, key=lambda x: -len(x))

    return cc

def largest_connected_set(C):
    r"""Compute connected components for a directed graph with weights
    represented by the given count matrix.

    Parameters
    ----------
    C : scipy.sparse matrix 
        Count matrix specifying edge weights.

    Returns
    -------
    lcc : array of integers
        The largest connected component of the directed graph.

    """
    return connected_sets(C)[0]

def connected_count_matrix(C):
    r"""Compute the count matrix of the largest connected set.

    The input count matrix is used as a weight matrix for the
    construction of a directed graph. The largest connected set of the
    constructed graph is computed. Vertices belonging to the largest
    connected component are used to generate a completely connected
    subgraph. The weight matrix of the subgraph is the desired
    completely connected count matrix.
    
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
    lcc=largest_connected_set(C)
    
    """Row slicing"""
    C_cc=C.tocsr()
    C_cc=C_cc[lcc, :]

    """Column slicing"""
    C_cc=C_cc.tocsc()
    C_cc=C_cc[:, lcc]

    return C_cc.tocoo()

    
