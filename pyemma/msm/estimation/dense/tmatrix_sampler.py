
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

__author__ = 'noe'

import numpy as np

def _sample_nonrev_single(C):
    # number of states
    n = C.shape[0]
    # target
    P = np.zeros((n,n), dtype = np.float64)
    # sample rows
    for i in range(n):
        # sample nonzeros only
        I = np.where(C[i] > 0)[0]
        # sample row with dirichlet
        pI = np.random.dirichlet(C[i,I][:]+1)
        # copy into relevant columns
        P[i,I] = pI[:]
    # done
    return P

def sample_nonrev(C, nsample=1):
    """ Generates a sample transition probability matrix given the count matrix C using dirichlet distributions

    Parameters
    ----------
    C : ndarray (n,n)
        count matrix

    Returns
    -------
    P : ndarray (n,n)
        independent sample of a transition matrix with respect to the likelihood p(C|P)

    """
    # copy C
    C = np.array(C)
    if nsample==1:
        return _sample_nonrev_single(C)
    elif nsample > 1:
        res = np.empty((nsample), dtype=object)
        for i in range(nsample):
            res[i] = _sample_nonrev_single(C)
        return res
    else:
        raise ValueError('nsample must be a positive integer')


class TransitionMatrixSampler:
    def __init__(self):
        pass

    def sample(self, nsample = 1):
        """

        Returns
        -------
        P : ndarray (n,n)
            sample of a transition matrix with respect to the likelihood p(C|P)

        """
        pass


class TransitionMatrixSamplerNonrev(TransitionMatrixSampler):
    def __init__(self, C):
        TransitionMatrixSampler.__init__(self)
        #super(TransitionMatrixSamplerNonrev, self).__init__()
        self._C = C

    def sample(self, nsample = 1):
        """

        Returns
        -------
        P : ndarray (n,n)
            sample of a transition matrix with respect to the likelihood p(C|P)

        """
        return sample_nonrev(self._C, nsample=nsample)