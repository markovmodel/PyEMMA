# This file is part of thermotools.
#
# Copyright 2015 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# thermotools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

r"""
This module provides functions to deal with connected sets.
"""

__all__ = [
    'compute_csets',
    'restrict_to_csets',
    'relabel_states']


import numpy as _np
import scipy as _sp
import scipy.sparse
import msmtools as _msmtools

def compute_csets(state_counts, count_matrices, nn=None):
    r'''
    Computes the largest connected sets for TRAM data.

    Parameters
    ----------
    state_counts : ndarray((T,M))
        Number of visits to the combiantions of thermodynamic state
        t and Markov state m
    count_matrices : ndarray((T,M,M))
        Count matrices for all T thermodynamic states.
    nn : int or None
        Number of neighbors that are assumed to overlap for the
        case of Umbrella sampling data.

        If n is None, all thermodynamic states are assumed to 
        overlap. The connected set is then computed by summing
        the count matrices over all thermodynamic states and
        taking it's largest strongly connected set.

        If nn is zero, the dynamics within every thermodynamic
        state is restricted to that state's largest connected
        sets. This is a very strong restriction and might remove
        interesting transitions from the data.

        If nn is not None and > 0, assume that the data comes 
        from an Umbrella sampling simmualtion and the number of
        the thermodynamic state matches the position of the 
        Umbrella along the order parameter. The connected set
        is computed by assuming that only Umbrellas up to the 
        nn'th neighbor (along the order parameter) overlap. 
        Technically this is computed by building an adjacency
        matrix on the product space of thermodynamic states and
        conformational states. The largest strongly connected set
        of that adjacency matrix determines the TRAM connted sets.
        In the matrix, the links within each thermodynamic state 
        (between different conformationals states) are just copied
        from the count matrices. The links between different
        thermodynamic states (within the same conformational
        state) are set according to the value of nn; if there are
        samples in both states (k,n) and (l,n) and |l-n|<=nn, a
        bidirectional link is added.

        nn = K-1 and nn = None should give the same result.

    Returns
    -------
    csets, projected_cset
    csets : list of ndarrays((X_i,), dtype=int)
        List indexed by thermodynamic state. Every element is the
        largest connected set of thermodynamic state k.
    projected_cset : ndarray(M, dtype=int)
        The overall connected set. This is the union of the 
        individual connected set of the thermodynamic states.
        It is useful for relabeling states in order to compress
        the data a bit while keeping the data structures
        non-ragged.
    '''
    n_therm_states, n_conf_states = state_counts.shape    

    if nn is None:
        # assume _direct_ overlap between all umbrellas
        C_sum = count_matrices.sum(axis=0)
        cset_projected = _msmtools.estimation.largest_connected_set(C_sum, directed=True)

        csets = []
        for k in range(n_therm_states):
            cset = _np.intersect1d(_np.where(state_counts[k, :]>0), cset_projected)
            csets.append(cset)

        return csets, cset_projected
    elif nn == 0:
        # within every thermodynamic state, restrict count this state's 
        # largest connected set
        csets = []
        C_sum = _np.zeros((n_conf_states, n_conf_states), dtype=count_matrices.dtype)
        for k in range(n_therm_states):
            cset = _msmtools.estimation.largest_connected_set(count_matrices[k,:,:], directed=True)
            csetT = cset[:, _np.newaxis]
            C_sum[csetT, cset] += count_matrices[k, csetT, cset]
            csets.append(cset)

        projected_cset = _msmtools.estimation.largest_connected_set(C_sum, directed=True)

        if len(_msmtools.estimation.connected_sets(C_sum, directed=True)) > 1:
            # if C_sum contained more than one strongly connected component,
            # restrict the individual csets of the thermodynamic state to
            # the largest one
            for k in range(n_therm_states):
                csets[k] = _np.intersect1d(csets[k], projected_cset)

        return csets, projected_cset
    else:
        assert nn>=1 and nn<=n_therm_states-1
        # assume overlap between nn neighboring umbrellas
        dim = n_therm_states*n_conf_states
        i_s = []
        j_s = []
        # connectivity between thermodynamic states
        for l in range(1,nn + 1):
            for k in range(n_therm_states - l):
                    w = _np.where(_np.logical_and(state_counts[k, :]>0, state_counts[k + l, :]>0))[0] 
                    a = w + k*n_conf_states
                    b = w + (k + l)*n_conf_states
                    i_s += list(a) # bi
                    j_s += list(b) # di
                    i_s += list(b) # rec
                    j_s += list(a) # tional
        # connectivity between conformational states:
        # just copy it from the count matrices
        for k in range(n_therm_states):
            temp = _sp.sparse.coo_matrix(count_matrices[k, :, :])
            i_s += list(temp.row + k*n_conf_states)
            j_s += list(temp.col + k*n_conf_states)

        data = _np.ones(len(i_s), dtype=int)
        A = _sp.sparse.coo_matrix((data, (i_s, j_s)), shape=(dim, dim))
        cset = _msmtools.estimation.largest_connected_set(A, directed=True)

        # group by thermodynamic state
        cset = _np.unravel_index(cset, (n_therm_states, n_conf_states), order='C')
        csets = [[] for k in range(n_therm_states)]
        for k,i in zip(*cset):
            csets[k].append(i)

        csets = [_np.array(c) for c in csets]
        projected_cset = _np.unique(_np.concatenate(csets))

        return csets, projected_cset

def restrict_to_csets(state_counts, count_matrices, tramtraj, csets):
    r'''
    Delete elements that are not in the connected set.

    Parameters
    ----------
    state_counts : ndarray((T,M))
        Number of visits to the combiantions of thermodynamic state
        t and Markov state m
    count_matrices : ndarray((T,M,M))
        Count matrices for all T thermodynamic states.
    tramtraj : ndarray((X,2+T))
        trajectory in TRAM format
    csets : list of ndarray((X_i),dtype=int), length=T
        List of connected sets for every thermodynamic state k.

    Returns
    -------
    state_counts, count_matrices, tramtraj 
    state_counts and count_matrices are in the same format
    as the input parameters. Elements not in the connected
    sets are zero. (To remove zero columns and rows, use 
    `relabel_states`)
    tram_traj : ndarray((Y,2+T))
    Same as input but with frames removed where the combination
    of thermodynamic state and Markov state is not in the 
    connected sets.
    '''
    new_state_counts = _np.zeros_like(state_counts, order='C', dtype=_np.intc)
    for k,cset in enumerate(csets):
        new_state_counts[k, cset] = state_counts[k, cset]

    new_count_matrices = _np.zeros_like(count_matrices, order='C', dtype=_np.intc)
    for k,cset in enumerate(csets):
        csetT = cset[:, _np.newaxis]
        new_count_matrices[k, csetT, cset] = count_matrices[k, csetT, cset]

    n_therm_states, n_conf_states = state_counts.shape
    valid = _np.zeros((n_therm_states, n_conf_states), dtype=bool)
    for k,cset in enumerate(csets):
        valid[k,cset] = True
    therm_state_traj = tramtraj[:, 0].astype(int)
    conf_state_traj = tramtraj[:, 1].astype(int)
    ok_traj = valid[therm_state_traj, conf_state_traj]
    new_tramtraj = tramtraj[ok_traj, :]

    return new_state_counts, new_count_matrices, new_tramtraj

def relabel_states(state_counts, count_matrices, dtraj, projected_cset):
    r'''
    Relabel states and remove zero columns/rows.

    Parameters
    ----------
    state_counts : ndarray((T,M))
        Number of visits to the combiantions of thermodynamic state
        t and Markov state m
    count_matrices : ndarray((T,M,M))
        Count matrices for all T thermodynamic states.
    dtraj : ndarray((X,))
        trajectory of Markov states. If the trajectory contains
        states that are not in the connected set, an exception
        is raised.
    projected_cset : ndarray((Y,))
        Union of the connected sets of all thermodynamic states.

    Returns
    -------
    state_counts and count_matrices are reduced in dimensions
    along the Markov state axes. Zero rows/columns are removed.
    dtraj is unchanged in dimension, but the Markov states have
    been relabeled.
    '''
    new_state_counts = state_counts[:, projected_cset]
    new_state_counts = _np.require(new_state_counts, dtype=_np.intc, requirements=['C', 'A'])

    new_count_matrices = count_matrices[:, projected_cset[:, _np.newaxis], projected_cset]
    new_count_matrices = _np.require(new_count_matrices, dtype=_np.intc, requirements=['C', 'A'])

    mapping = _np.ones(max(_np.max(projected_cset), _np.max(dtraj)) + 1, dtype=int)*(-1)
    mapping[projected_cset] = _np.arange(len(projected_cset), dtype=_np.intc)
    new_dtraj = mapping[dtraj]
    assert _np.all(new_dtraj != -1)

    return new_state_counts, new_count_matrices, new_dtraj
