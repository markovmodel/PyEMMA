
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import mdtraj as md
from mdtraj.formats import XTCTrajectoryFile
from pyemma.util.annotators import deprecated

__all__ = ['regroup_RAM',
           'regroup_DISK',
           'PCCA_disctrajs',
           'from_lcc_labels',
           'to_lcc_labels']


class MapToConnectedStateLabels():
    def __init__(self, lcc):
        r"""Mapping from original labels to lcc-indices.
        
        Parameters
        ----------
        lcc : (M, ) ndarray
            Largest connected set of microstates (original labels)
            
        """
        self.lcc = lcc
        self.new_labels = np.arange(len(self.lcc))
        self.dictmap = dict(zip(self.lcc, self.new_labels))

    def map(self, A):
        r"""Map subset of microstates to subset of connected
        microstates.

        Parameters
        ----------
        A : list of int
            Subset of microstate labels

        Returns
        -------
        A_cc : list of int
            Corresponding subset of mircrostate labels
            in largest connected set lcc. 

        """
        if not set(A).issubset(set(self.lcc)):
            raise ValueError("A is not a subset of the set of completely connected states.")
        else:
            return [self.dictmap[i] for i in A]


def from_lcc_labels(states_in_lcc, lcc):
    r"""Recover original labels of microstate after they have been
    relabeled by restricting them to the largest connected set.
    
    Parameters
    ----------
    states_in_lcc : array-like
         Array of microstates after relabeling (new labels)
    lcc : (M, ) ndarray
         Largest connected set of microstates (original labels)
         
    Returns
    -------
    states : array-like
        Array of microstates with original label
        
    """
    return lcc[states_in_lcc]


def to_lcc_labels(states, lcc):
    r"""Relabel microstates replacing the original label by the
    corresponding index in the largest connected set.
    
    Parameters
    ----------
    states : array-like
         Original microstate labels (original labels)
    lcc : (M, ) ndarray
         Largest connected set of microstates (original label)
         
    Returns
    -------
    states_in_lcc : array-like
         Relabeled microstates
    
    """
    mymap = MapToConnectedStateLabels(lcc)
    return mymap.map(states)


@deprecated("Please use pyemma.coordinates.save_trajs")
def regroup_RAM(trajs, disctrajs):
    r"""Regroups MD trajectories into clusters according to discretised trajectories.

    Parameters
    ----------
    trajs : list of `mdtraj.Trajectory`s
    disctrajs : list of array-likes
        len(disctrajs[i])==trajs[i].n_frames for all i

    Returns
    -------
    cluster : list of `mdtraj.Trajectory`s or `None`, len(cluster)=np.max(trajs)+1
       each element cluster[i] is either `None` if i wasn't found in disctrajs or
       is a new trajectory that holds all frames that were assigned to cluster i.

    Notes
    -----
    This function is RAM intensive.

    """

    # handle single element invocation
    if not isinstance(trajs, list):
        trajs = [trajs]
    if not isinstance(disctrajs, list):
        disctrajs = [disctrajs]

    assert len(disctrajs) == len(trajs), 'Number of disctrajs and number of trajs doesn\'t agree.'
    states = np.unique(np.hstack(([np.unique(disctraj) for disctraj in disctrajs])))
    states = np.setdiff1d(states, [-1])  # exclude invalid states
    cluster = [None] * (np.max(states) + 1)
    for disctraj, traj, i in zip(disctrajs, trajs, xrange(len(trajs))):
        assert len(disctraj) == traj.n_frames, 'Length of disctraj[%d] doesn\'t match number of frames in traj[%d].' % (
            i, i)
        for s in states:
            match = (disctraj == s)
            if np.count_nonzero(match) > 0:
                if cluster[s] is None:
                    cluster[s] = traj.xyz[match, :, :]
                else:
                    cluster[s] = np.concatenate((cluster[s], traj.xyz[match, :, :]), axis=0)
    for i in xrange(len(cluster)):
        if not cluster[i] is None:
            cluster[i] = md.Trajectory(cluster[i], trajs[0].topology)
    return cluster


def _regroup_DISK_subset(states, trajs, topology_file, disctrajs, path, stride):
    writer = [None] * (max(states) + 1)
    out_fnames = []
    for i in states:
        out_fname = path + os.sep + ('%d.xtc' % i)
        out_fnames.append(out_fname)
        writer[i] = XTCTrajectoryFile(out_fname, 'w', force_overwrite=True)

    for disctraj, traj in zip(disctrajs, trajs):
        reader = md.iterload(traj, top=topology_file, stride=stride)
        start = 0
        for chunk in reader:
            chunk_length = chunk.xyz.shape[0]
            for i in xrange(chunk_length):
                cl = disctraj[i + start]
                if cl in states:
                    writer[cl].write(chunk.xyz[i, :, :])
            start += chunk_length
    for i in states:
        writer[i].close()

    return out_fnames


def regroup_DISK(trajs, topology_file, disctrajs, path, stride=1, max_writers=100):
    """Regroups MD trajectories into clusters according to discretised trajectories.

    Parameters
    ----------
    trajs : list of strings 
        xtc/dcd/... trajectory file names
    topology_file : string
        name of topology file that matches `trajs`
    disctrajs : list of array-likes
        discretized trajectories
    path : string
        file system path to directory where cluster trajectories are written
    stride : int
        stride of disctrajs with respect to the (original) trajs
    max_writers : int, optional, default = 100
        maximum number of mdtraj trajectory writers to allocate at once

    Returns
    -------
    cluster : list of file names or `None`, len(cluster)=np.max(trajs)+1
        each element cluster[i] is either `None` if i wasn't found in disctrajs or
        is a the file name of a new trajectory that holds all frames that were 
        assigned to cluster i.
    """
    # handle single element invocation
    if not isinstance(trajs, list):
        trajs = [trajs]
    if not isinstance(disctrajs, list):
        disctrajs = [disctrajs]

    states = np.unique(np.hstack(([np.unique(disctraj) for disctraj in disctrajs])))
    states = np.setdiff1d(states, [-1])  # exclude invalid states
    
    out_fnames = []
    # break list of states into smaller list of a maximal length = max_writers 
    for i in xrange((len(states) - 1)//max_writers + 1):
        start = i*max_writers
        stop = start+max_writers
        states_subset = states[start:stop]
        out_fnames_subset = _regroup_DISK_subset(states_subset, trajs, topology_file, disctrajs, path, stride)
        out_fnames += out_fnames_subset
        
    return out_fnames


@deprecated("Please use pyemma.coordinates.save_trajs")
def PCCA_disctrajs(disctrajs, connected_set, memberships):
    r"""Compute disctrajs coarse-grained to the PCCA sets.

    Parameters
    ---------
    disctrajs : list of array-likes
        discretzed trajectories
    connected_set : (N) ndarray 
        connected set as returned by `pyemma.msm.estimation.largest_connected_set`
    memberships : (N,M) ndarray
        PCCA memberships as returned by `pyemma.msm.analysis.pcca`

    Returns
    -------
    cgdisctraj : list of array likes in the same shape as parameter `disctrajs`
    If disctaj[i][j] was assigned to PCCA set k, then cgdisctraj[i][j]==k.
    If disctaj[i][j] isn't in the connected set, cgdisctraj[i][j]==-1.
    """

    if not isinstance(disctrajs, list):
        disctrajs = [disctrajs]

    assert connected_set.ndim == 1
    assert connected_set.shape[0] == memberships.shape[0]

    # compute the forward map : old index -> new index
    backward_map = connected_set  # map : new index -> old index
    n_states = np.max(disctrajs) + 1
    forward_map = np.ones(n_states, dtype=int) * (-1)
    forward_map[backward_map] = np.arange(backward_map.shape[0])  # forward(backward)=Identity
    pcca_map = np.hstack((np.argmax(memberships, axis=1), [-1]))
    return [pcca_map[forward_map[d]] for d in disctrajs]
