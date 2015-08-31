#    Copyright (C) 2004-2015 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
import numpy as np

# taken from networkx.drawing.layout and added hold_dim
def _fruchterman_reingold(A, dim=2, k=None, pos=None, fixed=None,
                          iterations=50, hold_dim=None):
    # Position nodes in adjacency matrix A using Fruchterman-Reingold
    # Entry point for NetworkX graph is fruchterman_reingold_layout()
    try:
        nnodes, _ = A.shape
    except AttributeError:
        raise RuntimeError(
            "fruchterman_reingold() takes an adjacency matrix as input")

    A = np.asarray(A)  # make sure we have an array instead of a matrix

    if pos is None:
        # random initial positions
        pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
    else:
        # make sure positions are of same type as matrix
        pos = pos.astype(A.dtype)

    # optimal distance between nodes
    if k is None:
        k = np.sqrt(1.0 / nnodes)
    # the initial "temperature"  is about .1 of domain area (=1x1)
    # this is the largest step allowed in the dynamics.
    t = 0.1
    # simple cooling scheme.
    # linearly step down by dt on each iteration so last iteration is size dt.
    dt = t / float(iterations + 1)
    delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
    # the inscrutable (but fast) version
    # this is still O(V^2)
    # could use multilevel methods to speed this up significantly
    for _ in range(iterations):
        # matrix of difference between points
        for i in range(pos.shape[1]):
            delta[:, :, i] = pos[:, i, None] - pos[:, i]
        # distance between points
        distance = np.sqrt((delta**2).sum(axis=-1))
        # enforce minimum distance of 0.01
        distance = np.where(distance < 0.01, 0.01, distance)
        # displacement "force"
        displacement = np.transpose(np.transpose(delta) *
                                    (k * k / distance**2 - A * distance / k))\
            .sum(axis=1)
        # update positions
        length = np.sqrt((displacement**2).sum(axis=1))
        length = np.where(length < 0.01, 0.1, length)
        delta_pos = np.transpose(np.transpose(displacement) * t / length)
        if fixed is not None:
            # don't change positions of fixed nodes
            delta_pos[fixed] = 0.0
        # only update y component
        if hold_dim == 0:
            pos[:, 1] += delta_pos[:, 1]
        # only update x component
        elif hold_dim == 1:
            pos[:, 0] += delta_pos[:, 0]
        else:
            pos += delta_pos
        # cool temperature
        t -= dt
        pos = _rescale_layout(pos)
    return pos


def _rescale_layout(pos, scale=1):
    # rescale to (0,pscale) in all axes

    # shift origin to (0,0)
    lim = 0  # max coordinate for all axes
    for i in range(pos.shape[1]):
        pos[:, i] -= pos[:, i].min()
        lim = max(pos[:, i].max(), lim)
    # rescale to (0,scale) in all directions, preserves aspect
    for i in range(pos.shape[1]):
        pos[:, i] *= scale / lim
    return pos