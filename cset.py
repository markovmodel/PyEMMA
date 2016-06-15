# This file is part of thermotools.
#
# Copyright 2015, 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
    'restrict_to_csets']

import numpy as _np
import scipy as _sp
import scipy.sparse as _sps
import msmtools as _msmtools
import bar as _bar
import util as _util
import collections as _collections

def compute_csets_TRAM(
    connectivity, state_counts, count_matrices, equilibrium_state_counts=None,
    ttrajs=None, dtrajs=None, bias_trajs=None, nn=None, factor=1.0, callback=None):
    r"""
    Computes the largest connected sets for TRAM data.

    Parameters
    ----------
    connectivity : string
        one of 'summed_count_matrix', 'strong_in_every_ensemble',
        'neighbors', 'post_hoc_RE' or 'BAR_variance'
        Selects the algorithm for measuring overlap between thermodynamic
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
        matrix determines the TRAM connected sets. In the matrix, the
        links within each thermodynamic state (between different
        conformational states) are just copied from the count matrices.
        The links between different thermodynamic states (within the
        same conformational state) are set according to the value of nn;
        if there are samples in both states (k,n) and (l,n) and
        |l-n|<=nn, a bidirectional link is added.

        post_hoc_RE: like neighbors but don't assume any neighborhood
        relations between ensembles but compute them. A combination
        (i,k) of configuration state i and thermodynamic state k
        overlaps with (i,l) if a replica exchange simulation [1]_
        restricted to state i would show at least one transition from k
        to l or one transition from from l to k.
        The parameters ttrajs, dtrajs, bias_trajs must be set.

        BAR_variance: like neighbors but compute overlap between
        thermodynamic states using the BAR variance [2]_. Two states (i,k)
        and (i,l) overlap if the variance of the free energy difference
        \Delta f_{kl} (restricted to conformational state i) is less or
        equal than one.
        The parameter ttrajs, dtrajs, bias_trajs must be set.

    state_counts : numpy.ndarray((T, M), dtype=numpy.intc)
        Number of visits to the combinations of thermodynamic state t
        and Markov state m
    count_matrices : numpy.ndarray((T, M, M), dtype=numpy.intc)
        Count matrices for all T thermodynamic states.
    equilibrium_state_counts : numpy.dnarray((T, M)), optional
        Number of visits to the combinations of thermodynamic state t
        and Markov state m in the equilibrium data (for use with TRAMMBAR).
    ttrajs : list of numpy.ndarray(X_i, dtype=numpy.intc), optional
        List of generating thermodynamic state trajectories.
    dtrajs : list of numpy.ndarray(X_i, dtype=numpy.intc), optional
        List of configurational state trajectories (disctrajs).
    bias_trajs : list of numpy.ndarray((X_i, T), dtype=numpy.float64), optional
        List of bias energy trajectories.
        The last three parameters are only required for
        connectivity = 'post_hoc_RE' or connectivity = 'BAR_variance'.
    nn : int, optional
        Number of neighbors that are assumed to overlap when
        connectivity='neighbors'
    factor : int, default=1.0
        scaling factor used for connectivity = 'post_hoc_RE' or
        'BAR_variance'. Values greater than 1.0 weaken the connectivity
        conditions. For 'post_hoc_RE' this multiplies the number of
        hypothetically observed transtions. For 'BAR_variance' this
        scales the threshold for the minimal allowed variance of free
        energy differences.

    Returns
    -------
    csets, projected_cset
    csets : list of ndarrays((X_i,), dtype=int)
        List indexed by thermodynamic state. Every element csets[k] is
        the largest connected set at thermodynamic state k.
    projected_cset : ndarray(M, dtype=int)
        The overall connected set. This is the union of the individual
        connected sets of the thermodynamic states.

    References:
    -----------
    [1]_ Hukushima et al, Exchange Monte Carlo method and application to spin
    glass simulations, J. Phys. Soc. Jan. 65, 1604 (1996)
    [2]_ Shirts and Chodera, Statistically optimal analysis of samples
    from multiple equilibrium states, J. Chem. Phys. 129, 124105 (2008)
    """
    return _compute_csets(
        connectivity, state_counts, count_matrices, ttrajs, dtrajs, bias_trajs,
        nn=nn, equilibrium_state_counts=equilibrium_state_counts,
        factor=factor, callback=callback)

def compute_csets_dTRAM(connectivity, count_matrices, nn=None, callback=None):
    r"""
    Computes the largest connected sets for dTRAM data.

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

    count_matrices : numpy.ndarray((T, M, M))
        Count matrices for all T thermodynamic states.
    nn : int or None, optional
        Number of neighbors that are assumed to overlap when
        connectivity='neighbors'

    Returns
    -------
    csets, projected_cset
    csets : list of numpy.ndarray((M_prime_k,), dtype=int)
        List indexed by thermodynamic state. Every element csets[k] is
        the largest connected set at thermodynamic state k.
    projected_cset : numpy.ndarray(M_prime, dtype=int)
        The overall connected set. This is the union of the individual
        connected sets of the thermodynamic states.
    """
    if connectivity=='post_hoc_RE' or connectivity=='BAR_variance':
        raise Exception('Connectivity type %s not supported for dTRAM data.'%connectivity)
    state_counts =  _np.maximum(count_matrices.sum(axis=1), count_matrices.sum(axis=2))
    return _compute_csets(
        connectivity, state_counts, count_matrices, None, None, None, nn=nn, callback=callback)

def _overlap_BAR_variance(a, b, factor=1.0):
    N_1 = a.shape[0]
    N_2 = b.shape[0]
    db_IJ = _np.zeros(N_1, dtype=_np.float64)
    db_JI = _np.zeros(N_2, dtype=_np.float64)
    db_IJ[:] = a[:, 1] - a[:, 0]
    db_JI[:] = b[:, 0] - b[:, 1]
    df = _bar.df(db_IJ, db_JI, _np.zeros(N_1 + N_2, dtype=_np.float64))
    u = _np.concatenate((a, b), axis=0)
    du = u[:, 1] - u[:, 0]
    b = (1.0 / (2.0 + 2.0 * _np.cosh(df - du - _np.log(1.0 * N_1 / N_2)))).sum()
    return (1 / b - (N_1 + N_2) / (N_1 * N_2)) < factor

def _compute_csets(
    connectivity, state_counts, count_matrices, ttrajs, dtrajs, bias_trajs, nn,
    equilibrium_state_counts=None, factor=1.0, callback=None):
    n_therm_states, n_conf_states = state_counts.shape

    if equilibrium_state_counts is not None:
        all_state_counts = state_counts + equilibrium_state_counts
    else:
        all_state_counts = state_counts

    if connectivity == 'summed_count_matrix':
        # assume _direct_ overlap between all umbrellas
        C_sum = count_matrices.sum(axis=0)
        if equilibrium_state_counts is not None:
            eq_states = _np.where(equilibrium_state_counts.sum(axis=0) > 0)[0]
            C_sum[eq_states, eq_states[:, _np.newaxis]] = 1
        cset_projected = _msmtools.estimation.largest_connected_set(C_sum, directed=True)
        csets = []
        for k in range(n_therm_states):
            cset = _np.intersect1d(_np.where(all_state_counts[k, :] > 0), cset_projected)
            csets.append(cset)
        return csets, cset_projected
    elif connectivity == 'strong_in_every_ensemble':
        # within every thermodynamic state, restrict counts to this state's
        # largest connected set
        if equilibrium_state_counts is not None:
            raise Exception('Connectivity mode "strong_in_every_ensemble" doesn\'t yet support equilibrium data.')
        csets = []
        C_sum = _np.zeros((n_conf_states, n_conf_states), dtype=count_matrices.dtype)
        for k in range(n_therm_states):
            if callback is not None:
                callback(maxiter=n_therm_states, iteration_step=k)
            cset = _msmtools.estimation.largest_connected_set(
                count_matrices[k, :, :], directed=True)
            csetT = cset[:, _np.newaxis]
            C_sum[csetT, cset] += count_matrices[k, csetT, cset]
            csets.append(cset)
        projected_cset = _msmtools.estimation.largest_connected_set(C_sum, directed=True)
        if len(_msmtools.estimation.connected_sets(C_sum, directed=True)) > 1:
            # if C_sum contained more than one strongly connected component,
            # restrict the individual csets of the thermodynamic states to
            # the largest one
            for k in range(n_therm_states):
                csets[k] = _np.intersect1d(csets[k], projected_cset)
        return csets, projected_cset
    elif connectivity in ['neighbors', 'post_hoc_RE', 'BAR_variance']:
        dim = n_therm_states * n_conf_states
        if connectivity == 'post_hoc_RE' or connectivity == 'BAR_variance':
            if connectivity == 'post_hoc_RE':
                overlap = _util._overlap_post_hoc_RE
            else:
                overlap = _overlap_BAR_variance
            i_s = []
            j_s = []
            for i in range(n_conf_states):
                # can take a very long time, allow to report progress via callback
                if callback is not None:
                    callback(maxiter=n_conf_states, iteration_step=i)
                therm_states = _np.where(all_state_counts[:, i] > 0)[0] # therm states that have samples
                # prepare list of indices for all thermodynamic states
                traj_indices = {}
                frame_indices = {}
                for k in therm_states:
                    frame_indices[k] = [_np.where(
                        _np.logical_and(d == i, t == k))[0] for t, d in zip(ttrajs, dtrajs)]
                    traj_indices[k] = [j for j, fi in enumerate(frame_indices[k]) if len(fi) > 0]
                for k in therm_states:
                    for l in therm_states:
                        if k!=l:
                            kl = _np.array([k, l])
                            a = _np.concatenate([
                                bias_trajs[j][:, kl][frame_indices[k][j], :] for j in traj_indices[k]])
                            b = _np.concatenate([
                                bias_trajs[j][:, kl][frame_indices[l][j], :] for j in traj_indices[l]])
                            if overlap(a, b, factor=factor):
                                x = i + k * n_conf_states
                                y = i + l * n_conf_states
                                i_s.append(x)
                                j_s.append(y)
                                i_s.append(y)
                                j_s.append(x)
        else: # assume overlap between nn neighboring umbrellas
            assert nn is not None, 'With connectivity="neighbors", nn can\'t be None.'
            assert nn >= 1 and nn <= n_therm_states - 1
            i_s = []
            j_s = []
            # connectivity between thermodynamic states
            for l in range(1, nn + 1):
                if callback is not None:
                    callback(maxiter=nn, iteration_step=l)
                for k in range(n_therm_states - l):
                    w = _np.where(_np.logical_and(
                        all_state_counts[k, :] > 0, all_state_counts[k + l, :] > 0))[0]
                    a = w + k * n_conf_states
                    b = w + (k + l) * n_conf_states
                    i_s += list(a) # bi
                    j_s += list(b) # di
                    i_s += list(b) # rec
                    j_s += list(a) # tional

        # connectivity between conformational states:
        # just copy it from the count matrices
        for k in range(n_therm_states):
            temp = _sp.sparse.coo_matrix(count_matrices[k, :, :])
            i_s += list(temp.row + k * n_conf_states)
            j_s += list(temp.col + k * n_conf_states)

        # If there is global equilibrium data, assume full connectivity
        # between all visited conformational states within the same thermodynamic state.
        if equilibrium_state_counts is not None:
            for k in range(n_therm_states):
                vertices = _np.where(equilibrium_state_counts[k, :]>0)[0]
                # add bidirectional chain that links all states
                chain = (vertices[0:-1], vertices[1:])
                i_s += list(chain[0] + k * n_conf_states)
                j_s += list(chain[1] + k * n_conf_states)
                i_s += list(chain[1] + k * n_conf_states)
                j_s += list(chain[0] + k * n_conf_states)

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
        raise Exception(
            'Unknown value "%s" of connectivity. Should be one of: \
            summed_count_matrix, strong_in_every_ensemble, neighbors, \
            post_hoc_RE or BAR_variance.' % connectivity)

def restrict_to_csets(
    csets, state_counts=None, count_matrices=None, ttrajs=None, dtrajs=None, bias_trajs=None):
    r"""
    Delete or deactivate elements that are not in the connected sets.

    Parameters
    ----------
    csets : list of numpy.ndarray((M_i), dtype=int), length=T
        List of connected sets for every thermodynamic state t.
    state_counts : numpy.ndarray((T, M)), optional
        Number of visits to the combinations of thermodynamic state
        t and Markov state m.
    count_matrices : numpy.ndarray((T, M, M)), optional
        Count matrices for all T thermodynamic states.
    ttrajs : list of numpy.ndarray(X_i), optional
        List of generating thermodynamic state trajectories.
        Only needed if dtrajs or bias_trajs are given as well.
        Is used to determine which frames are in the connected sets.
    dtrajs : list of ndarray(X_i), optional
        List of configurational state trajectories (disctrajs).
        If given, ttrajs must be set as well.
    bias_trajs : list of ndarray((X_i, T)), optional
        List of bias energy trajectories for all T thermodynamic states.
        If given, ttrajs and dtrajs must be given as well.

    Returns
    -------
    Modified copies of:
    state_counts, count_matrices, dtrajs, bias_trajs

    state_counts, count_matrices and dtrajs are in the same format
    as the input parameters. Elements of state_counts and cont_matrices
    not in the connected sets are zero. Elements of dtrajs not in the
    connected sets are negative.

    bias_trajs : list of ndarray((Y_i, T))
    Same as input but with frames removed where the combination
    of thermodynamic state and Markov state as given in ttrajs and
    dtrajs is not in the connected sets.
    """
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
    if dtrajs is not None:
        assert ttrajs is not None, 'ttrajs can\'t be None, when dtrajs are given.'
        n_therm_states, n_conf_states = state_counts.shape
        invalid = _np.ones((n_therm_states, n_conf_states), dtype=bool)
        for k, cset in enumerate(csets):
            if len(cset) > 0:
                invalid[k, cset] = False
        new_dtrajs = []
        assert len(ttrajs) == len(dtrajs)
        for t, d in zip(ttrajs, dtrajs):
            assert len(t) == len(d)
            new_d = _np.array(d, dtype=_np.intc, copy=True, order='C', ndmin=1)
            bad = invalid[t, d]
            new_d[bad] = new_d[bad] - n_conf_states # 'numpy equivalent' indices as in x[i]==x[i-len(x)]
            assert _np.all(new_d[bad] - n_conf_states < 0)
            new_dtrajs.append(new_d)
    else:
        new_dtrajs = None
    if bias_trajs is not None:
        assert ttrajs is not None, 'ttrajs can\'t be None, when bias_trajs are given.'
        assert dtrajs is not None, 'dtrajs can\'t be None, when bias_trajs are given.'
        n_therm_states, n_conf_states = state_counts.shape
        valid = _np.zeros((n_therm_states, n_conf_states), dtype=bool)
        for k, cset in enumerate(csets):
            if len(cset) > 0:
                valid[k, cset] = True
        new_bias_trajs = []
        assert len(ttrajs) == len(dtrajs) == len(bias_trajs)
        for t, d, b in zip(ttrajs, dtrajs, bias_trajs):
            assert len(t) == len(d) == len(b)
            ok_traj = valid[t, d]
            new_b = _np.zeros((_np.count_nonzero(ok_traj), b.shape[1]), dtype=_np.float64)
            new_b[:] = b[ok_traj, :]
            new_bias_trajs.append(new_b)
    else:
        new_bias_trajs = None
    return new_state_counts, new_count_matrices, new_dtrajs, new_bias_trajs
