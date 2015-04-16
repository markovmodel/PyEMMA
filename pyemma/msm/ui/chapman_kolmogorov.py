
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

r"""Chapman-Kolmogorov-Test

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""
import warnings

import numpy as np
from scipy.sparse import issparse

from pyemma.msm.estimation import cmatrix, connected_cmatrix, largest_connected_set, tmatrix
from pyemma.msm.analysis import statdist
from pyemma.msm.analysis import pcca_sets

from mapping import MapToConnectedStateLabels

__all__ = ['cktest']


def propagate(W, P, n=1):
    r"""Propagate probabilities.

    Parameters
    ----------
    W : (K, M) ndarray
        Matrix of vectors
    P : (M, M) ndarray or scipy.sparse matrix
        Transition matrix
    n : int (optional)
        Number of steps

    Returns
    -------
    W_n : (K, M) ndarray
        Matrix of propagated vectors
        
    """
    if issparse(P):
        """Ensure csr format"""
        P = P.tocsr()
        """Transpose W and P"""
        WT = W.T
        PT = P.T
        for i in range(n):
            WT = PT.dot(WT)
        """Transpose propgated WT"""
        W_n = WT.T
    else:
        W_n = 1.0 * W
        for i in range(n):
            W_n = np.dot(W_n, P)
    return W_n


def cktest(T_MSM, lcc_MSM, dtrajs, lag, K, nsets=2, sets=None, full_output=False):
    r"""Perform Chapman-Kolmogorov tests for given data.

    Parameters
    ----------
    T_MSM : (M, M) ndarray or scipy.sparse matrix
        Transition matrix of estimated MSM
    lcc_MSM : array-like
        Largest connected set of the estimated MSM
    dtrajs : list
        discrete trajectories
    lag : int
        lagtime for the MSM estimation
    K : int 
        number of time points for the test
    nsets : int, optional
        number of PCCA sets on which to perform the test
    sets : list, optional
        List of user defined sets for the test
    full_output : bool, optional
        Return additional information about set_factors

    Returns
    -------
    p_MSM : (K, nsets) ndarray
        p_MSM[k, l] is the probability of making a transition from
        set l to set l after k*lag steps for the MSM computed at 1*lag
    p_MD : (K, nsets) ndarray
        p_MD[k, l] is the probability of making a transition from
        set l to set l after k*lag steps as estimated from the given data
    eps_MD : (K, nsets)
        eps_MD[k, l] is an estimate for the statistical error of p_MD[k, l]    
    set_factors : (K, nsets) ndarray, optional
        set_factor[k, i] is the quotient of the MD and the MSM set probabilities

    References
    ----------
    .. [1] Prinz, J H, H Wu, M Sarich, B Keller, M Senne, M Held, J D
        Chodera, C Schuette and F Noe. 2011. Markov models of
        molecular kinetics: Generation and validation. J Chem Phys
        134: 174105
        
    """
    p_MD = np.zeros((K, nsets))
    p_MSM = np.zeros((K, nsets))
    eps_MD = np.zeros((K, nsets))
    set_factors = np.zeros((K, nsets))

    if sets is None:
        """Compute PCCA-sets from MSM at lagtime 1*\tau"""
        if issparse(T_MSM):
            msg = ("Converting sparse transition matrix to dense\n"
                   "since PCCA is currently only implemented for dense matrices.\n"
                   "You can avoid automatic conversion to dense arrays by\n"
                   "giving sets for the Chapman-Kolmogorov test explicitly")
            warnings.warn(msg, UserWarning)
            sets = pcca_sets(T_MSM.toarray(), nsets)
        else:
            sets = pcca_sets(T_MSM, nsets)
    nsets = len(sets)

    # translate sets from connected-set indexes to full indexes, where the comparison is made:
    for i, s in enumerate(sets):
        sets[i] = lcc_MSM[s]

    """Stationary distribution at 1*tau"""
    mu_MSM = statdist(T_MSM)

    """Mapping to lcc at lagtime 1*tau"""
    lccmap_MSM = MapToConnectedStateLabels(lcc_MSM)

    """Compute stationary distribution on sets"""
    w_MSM_1 = np.zeros((nsets, mu_MSM.shape[0]))
    for l in range(nsets):
        A = sets[l]
        A_MSM = lccmap_MSM.map(A)
        w_MSM_1[l, A_MSM] = mu_MSM[A_MSM] / mu_MSM[A_MSM].sum()

    w_MSM_k = 1.0 * w_MSM_1

    p_MSM[0, :] = 1.0
    p_MD[0, :] = 1.0
    eps_MD[0, :] = 0.0
    set_factors[0, :] = 1.0

    for k in range(1, K):
        """Propagate probability vectors for MSM"""
        w_MSM_k = propagate(w_MSM_k, T_MSM)

        """Estimate model at k*tau and normalize to make 'uncorrelated'"""
        C_MD = cmatrix(dtrajs, k * lag, sliding=True) / (k * lag)
        lcc_MD = largest_connected_set(C_MD)
        Ccc_MD = connected_cmatrix(C_MD, lcc=lcc_MD)
        """State counts for MD"""
        c_MD = Ccc_MD.sum(axis=1)
        """Transition matrix at k*tau"""
        T_MD = tmatrix(Ccc_MD)

        """Mapping to lcc at lagtime k*tau"""
        lccmap_MD = MapToConnectedStateLabels(lcc_MD)

        """Intersection of lcc_1 and lcc_k. lcc_k is not necessarily contained within lcc_1"""
        lcc = np.intersect1d(lcc_MSM, lcc_MD)

        """Stationary distribution restricted to lcc at lagtime k*tau"""
        mu_MD = np.zeros(T_MD.shape[0])
        """Extract stationary values in 'joint' lcc and assining them to their respective position"""
        mu_MD[lccmap_MD.map(lcc)] = mu_MSM[lccmap_MSM.map(lcc)]

        """Obtain sets and distribution at k*tau"""
        w_MD_1 = np.zeros((nsets, mu_MD.shape[0]))
        for l in range(len(sets)):
            A = sets[l]
            """Intersect the set with the lcc at lagtime k*tau"""
            A_new = np.intersect1d(A, lcc)
            if A_new.size > 0:
                A_MD = lccmap_MD.map(A_new)
                w_MD_1[l, A_MD] = mu_MD[A_MD] / mu_MD[A_MD].sum()

        """Propagate vector by the MD model"""
        w_MD_k = propagate(w_MD_1, T_MD)

        """Compute values"""

        for l in range(len(sets)):
            A = sets[l]
            """MSM values"""
            A_MSM = lccmap_MSM.map(A)
            p_MSM[k, l] = w_MSM_k[l, A_MSM].sum()

            """MD values"""
            A_new = np.intersect1d(A, lcc)
            if A_new.size > 0:
                A_MD = lccmap_MD.map(A_new)
                prob_MD = w_MD_k[l, A_MD].sum()
                p_MD[k, l] = prob_MD
                """Statistical errors"""
                c = c_MD[A_MD].sum()
                eps_MD[k, l] = np.sqrt(k * (prob_MD - prob_MD ** 2) / c)
                set_factors[k, l] = mu_MSM[lccmap_MSM.map(A_new)].sum() / mu_MSM[A_MSM].sum()

    if full_output:
        return p_MSM, p_MD, eps_MD, set_factors
    else:
        return p_MSM, p_MD, eps_MD