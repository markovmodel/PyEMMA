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
    'compute_csets_TRAM',
    'compute_csets_dTRAM',
    'restrict_to_csets',
    'relabel_states']


import numpy as _np
import scipy as _sp
import scipy.sparse as _sps
import msmtools as _msmtools
import bar as _bar


def _overlap_post_hoc_RE(state_counts, tram_sequence, factor=1.0):
    edges_i = []
    edges_j = []
    n_conf_states = state_counts.shape[1]

    for i in range(n_conf_states):
        therm_states = _np.where(state_counts[:, i]>0)[0] # therm states that have samples
        # prepare list of indices for all thermodynamic states
        indices = {}
        for k in therm_states:
            indices[k] = _np.where(_np.logical_and(tram_sequence[:, 1]==i, tram_sequence[:, 0]==k))[0]
        for k in therm_states:
            for l in therm_states:
                if k!=l:
                    kl = _np.array([2+k, 2+l])
                    a = tram_sequence[indices[k],:][:, kl]
                    b = tram_sequence[indices[l],:][:, kl]
                    delta = a[_np.newaxis,:,0]+b[:,_np.newaxis,1]-a[_np.newaxis,:,1]-b[:,_np.newaxis,0]
                    n_avg = _np.mean(_np.minimum(_np.exp(delta), 1.0))
                    if min(len(a),len(b)) * n_avg * factor >= 1.0:
                        a = i + k*n_conf_states
                        b = i + l*n_conf_states
                        edges_i.append(a)
                        edges_j.append(b)
                        edges_i.append(b)
                        edges_j.append(a)

    return edges_i, edges_j

def _bar_variance(a, b):
    N_1 = a.shape[0]
    N_2 = b.shape[0]
    db_IJ = _np.zeros(N_1, dtype=_np.float64)
    db_JI = _np.zeros(N_2, dtype=_np.float64)
    db_IJ[:] = a[:,1]-a[:,0]
    db_JI[:] = b[:,0]-b[:,1]
    df = _bar.df(db_IJ, db_JI, _np.zeros(N_1+N_2, dtype=_np.float64))
    u = _np.concatenate((a,b), axis=0)
    du = u[:,1]-u[:,0]
    b = (1.0/(2.0 + 2.0*_np.cosh(df - du - _np.log(1.0*N_1/N_2)))).sum()
    return 1/b - (N_1+N_2)/(N_1*N_2)

def _overlap_BAR_variance(state_counts, tram_sequence, factor=1.0):
    edges_i = []
    edges_j = []
    n_conf_states = state_counts.shape[1]

    for i in range(n_conf_states):
        therm_states = _np.where(state_counts[:, i]>0)[0]
        indices = {}
        for k in therm_states:
            indices[k] = _np.where(_np.logical_and(tram_sequence[:, 1]==i, tram_sequence[:, 0]==k))[0]
        for k in therm_states:
            for l in therm_states:
                if k!=l:
                    kl = _np.array([2+k, 2+l])
                    a = tram_sequence[indices[k],:][:, kl]
                    b = tram_sequence[indices[l],:][:, kl]
                    if _bar_variance(a, b) < factor:
                        a = i + k*n_conf_states
                        b = i + l*n_conf_states
                        edges_i.append(a)
                        edges_j.append(b)
                        edges_i.append(b)
                        edges_j.append(a)

    return edges_i, edges_j

def compute_csets_TRAM(connectivity, state_counts, count_matrices, tram_trajs, nn=None, factor=1.0):
    r'''
    Computes the largest connected sets for TRAM data.

    Parameters
    ----------
    connectivity : string
        one of 'summed_count_matrix', 'strong_in_every_ensemble', 
        'neighbors', 'post_hoc_RE' or 'BAR_variance'
        Selects the algortihm for measuring overlap between thermodynamic
        states. 

        summed_count_matrix: all thermodynamic states are assumed to 
        overlap. The connected set is then computed by summing
        the count matrices over all thermodynamic states and
        taking it's largest strongly connected set.

        strong_in_every_ensemble: the dynamics within every thermodynamic
        state is restricted to that state's largest connected
        set. This is a very strong restriction and might remove
        interesting transitions from the data.

        neighbors: assume that the data comes from an Umbrella sampling
        simulation and the number of the thermodynamic state matches
        the position of the Umbrella along the order parameter. The
        connected set is computed by assuming that only Umbrellas up to
        the nn'th neighbor (along the order parameter) overlap. 
        Technically this is computed by building an adjacency matrix on
        the product space of thermodynamic states and conformational
        states. The largest strongly connected set of that adjacency
        matrix determines the TRAM connted sets. In the matrix, the
        links within each thermodynamic state (between different
        conformationals states) are just copied from the count matrices.
        The links between different thermodynamic states (within the
        same conformational state) are set according to the value of nn;
        if there are samples in both states (k,n) and (l,n) and
        |l-n|<=nn, a bidirectional link is added.

        post_hoc_RE: like neighbors but don't assume any neighborhood
        relations between ensembles but compute them. A combination
        (i,k) of configuration state i and thermodynamic state k
        overlaps with (i,l) if a replica exchange simulation restricted
        to state i would show at least one transition from k to l and
        one transition from from l to k.
        The parameter tram_trajs must be set.

        BAR_variance: like neighbors but compute overlap between 
        thermodynamic states using the BAR variance _[1]. Two states (i,k)
        and (i,l) overlap if the variance of the free energy difference
        \Delta f_{kl} is less or equal than one.
        The parameter tram_trajs must be set.

    state_counts : ndarray((T,M))
        Number of visits to the combiantions of thermodynamic state t 
        and Markov state m
    count_matrices : ndarray((T,M,M))
        Count matrices for all T thermodynamic states.
    tram_trajs : ndarray((X,2+T))
        a TRAM trajectory, only required for connectivity = 'post_hoc_RE'
        or connectivity = 'BAR_variance'.
    nn : int or None, optional
        Number of neighbors that are assumed to overlap when
        connectivity='neighbors'
    factor : int, default=1.0
        scaling factor used for connectivity = 'post_hoc_RE' or
        'BAR_variance'. Values greater than 1.0 weaken the connectivity
        conditions. For 'post_hoc_RE' this multiplies the number of
        hypothetically overseved transtions. For 'BAR_variance' this 
        scales the threshold for the minimal allowd variance of free
        energy differences.

    Returns
    -------
    csets, projected_cset
    csets : list of ndarrays((X_i,), dtype=int)
        List indexed by thermodynamic state. Every element is the largest
        connected set of thermodynamic state k.
    projected_cset : ndarray(M, dtype=int)
        The overall connected set. This is the union of the individual
        connected sets of the thermodynamic states. It is useful for
        relabeling states in order to compress the data a bit while
        keeping the data structures non-ragged.

    References:
    -----------
    _[1] Shirts and Chodera, Statistically optimal analysis of samples
    from multiple equilibrium states, J. Chem. Phys. 129, 124105 (2008)
    '''
    return _compute_csets(connectivity, state_counts, count_matrices, tram_trajs, nn=nn, factor=factor)

def compute_csets_dTRAM(connectivity, count_matrices, nn=None):
    r'''
    Computes the largest connected sets for TRAM data.

    Parameters
    ----------
    connectivity : string
        one of 'summed_count_matrix', 'strong_in_every_ensemble', 
        'neighbors'.
        Selects the algortihm for measuring overlap between thermodynamic
        states. 

        summed_count_matrix: all thermodynamic states are assumed to 
        overlap. The connected set is then computed by summing
        the count matrices over all thermodynamic states and
        taking it's largest strongly connected set.

        strong_in_every_ensemble: the dynamics within every thermodynamic
        state is restricted to that state's largest connected
        set. This is a very strong restriction and might remove
        interesting transitions from the data.

        neighbors: assume that the data comes from an Umbrella sampling
        simulation and the number of the thermodynamic state matches
        the position of the Umbrella along the order parameter. The
        connected set is computed by assuming that only Umbrellas up to
        the nn'th neighbor (along the order parameter) overlap. 
        Technically this is computed by building an adjacency matrix on
        the product space of thermodynamic states and conformational
        states. The largest strongly connected set of that adjacency
        matrix determines the TRAM connted sets. In the matrix, the
        links within each thermodynamic state (between different
        conformationals states) are just copied from the count matrices.
        The links between different thermodynamic states (within the
        same conformational state) are set according to the value of nn;
        if there are samples in both states (k,n) and (l,n) and
        |l-n|<=nn, a bidirectional link is added.

    count_matrices : ndarray((T,M,M))
        Count matrices for all T thermodynamic states.
    nn : int or None, optional
        Number of neighbors that are assumed to overlap when
        connectivity='neighbors'

    Returns
    -------
    csets, projected_cset
    csets : list of ndarrays((X_i,), dtype=int)
        List indexed by thermodynamic state. Every element is the largest
        connected set of thermodynamic state k.
    projected_cset : ndarray(M, dtype=int)
        The overall connected set. This is the union of the individual
        connected sets of the thermodynamic states. It is useful for
        relabeling states in order to compress the data a bit while
        keeping the data structures non-ragged.
    '''        
    if connectivity=='post_hoc_RE' or connectivity=='BAR_variance':
        raise Exception('Connectivity type %s not supported for dTRAM data.'%connectivity)

    state_counts =  _np.maximum(count_matrices.sum(axis=1), count_matrices.sum(axis=2))

    return _compute_csets(connectivity, state_counts, count_matrices, None, nn=nn)

def _compute_csets(connectivity, state_counts, count_matrices, tram_trajs, nn=None, factor=1.0):
    n_therm_states, n_conf_states = state_counts.shape    

    if connectivity == 'summed_count_matrix':
        # assume _direct_ overlap between all umbrellas
        C_sum = count_matrices.sum(axis=0)
        cset_projected = _msmtools.estimation.largest_connected_set(C_sum, directed=True)

        csets = []
        for k in range(n_therm_states):
            cset = _np.intersect1d(_np.where(state_counts[k, :]>0), cset_projected)
            csets.append(cset)

        return csets, cset_projected

    elif connectivity == 'strong_in_every_ensemble':
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

    elif connectivity in ['neighbors', 'post_hoc_RE', 'BAR_variance']:
        # assume overlap between nn neighboring umbrellas
        dim = n_therm_states*n_conf_states
        if connectivity == 'post_hoc_RE':
            i_s, j_s = _overlap_post_hoc_RE(state_counts, tram_trajs, factor=factor)
        elif connectivity == 'BAR_variance':
            i_s, j_s = _overlap_BAR_variance(state_counts, tram_trajs, factor=factor)
        else: # neighbors
            assert nn>=1 and nn<=n_therm_states-1
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

        csets = [_np.array(c,dtype=int) for c in csets]
        projected_cset = _np.unique(_np.concatenate(csets))

        return csets, projected_cset
    else:
        raise Exception('Unknown value "%s" of connectivity. Should be one of: \
                         summed_count_matrix, strong_in_every_ensemble, neighbors, \
                         post_hoc_RE or BAR_variance.' % connectivity)

def restrict_to_csets(state_counts, count_matrices, tramtraj, csets):
    r'''
    Delete elements that are not in the connected set.

    Parameters
    ----------
    state_counts : ndarray((T,M)), optional
        Number of visits to the combiantions of thermodynamic state
        t and Markov state m
    count_matrices : ndarray((T,M,M)), optional
        Count matrices for all T thermodynamic states.
    tramtraj : ndarray((X,2+T)), optional
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
    if state_counts is not None:
        new_state_counts = _np.zeros_like(state_counts, order='C', dtype=_np.intc)
        for k,cset in enumerate(csets):
            if len(cset)>0:
                new_state_counts[k, cset] = state_counts[k, cset]
    else:
        new_state_counts = None

    if count_matrices is not None:
        new_count_matrices = _np.zeros_like(count_matrices, order='C', dtype=_np.intc)
        for k,cset in enumerate(csets):
            if len(cset)>0:
                csetT = cset[:, _np.newaxis]
                new_count_matrices[k, csetT, cset] = count_matrices[k, csetT, cset]
    else:
        new_count_matrices = None

    if tramtraj is not None:
        n_therm_states, n_conf_states = state_counts.shape
        valid = _np.zeros((n_therm_states, n_conf_states), dtype=bool)
        for k,cset in enumerate(csets):
            if len(cset)>0:
                valid[k,cset] = True
        therm_state_traj = tramtraj[:, 0].astype(int)
        conf_state_traj = tramtraj[:, 1].astype(int)
        ok_traj = valid[therm_state_traj, conf_state_traj]
        new_tramtraj = tramtraj[ok_traj, :]
    else:
        new_tramtraj = None

    return new_state_counts, new_count_matrices, new_tramtraj

def relabel_states(state_counts, count_matrices, dtraj, projected_cset):
    r'''
    Relabel states and remove zero columns/rows.

    Parameters
    ----------
    state_counts : ndarray((T,M)), optional
        Number of visits to the combiantions of thermodynamic state
        t and Markov state m
    count_matrices : ndarray((T,M,M)), optional
        Count matrices for all T thermodynamic states.
    dtraj : ndarray((X,)), optional
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
    if state_counts is not None:
        new_state_counts = state_counts[:, projected_cset]
        new_state_counts = _np.require(new_state_counts, dtype=_np.intc, requirements=['C', 'A'])
    else:
        new_state_counts = None

    if count_matrices is not None:
        new_count_matrices = count_matrices[:, projected_cset[:, _np.newaxis], projected_cset]
        new_count_matrices = _np.require(new_count_matrices, dtype=_np.intc, requirements=['C', 'A'])
    else:
        new_count_matrices = None

    if dtraj is not None:
        mapping = _np.ones(max(_np.max(projected_cset), _np.max(dtraj)) + 1, dtype=int)*(-1)
        mapping[projected_cset] = _np.arange(len(projected_cset), dtype=_np.intc)
        new_dtraj = mapping[dtraj]
        assert _np.all(new_dtraj != -1)
    else:
        new_dtraj = None

    return new_state_counts, new_count_matrices, new_dtraj
