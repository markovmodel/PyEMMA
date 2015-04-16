
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

r"""Dense implementation of hitting probabilities

.. moduleauthor:: F.Noe <frank DOT noe AT fu-berlin DOT de>

"""

import numpy as np


def hitting_probability(P, target):
    """
    Computes the hitting probabilities for all states to the target states.
    
    The hitting probability of state i to set A is defined as the minimal, 
    non-negative solution of:
    
    .. math::
        h_i^A &= 1                    \:\:\:\:  i\in A \\
        h_i^A &= \sum_j p_{ij} h_i^A  \:\:\:\:  i \notin A
    
    Returns
    =======
    h : ndarray(n)
        a vector with hitting probabilities
    """
    if hasattr(target, "__len__"):
        target = np.array(target)
    else:
        target = np.array([target])
    # target size
    n = np.shape(P)[0]
    # nontarget
    nontarget = np.array(list(set(range(n)) - set(target)), dtype=int)
    # stable states
    stable = np.where(np.isclose(np.diag(P), 1) == True)[0]
    # everything else
    origin = np.array(list(set(nontarget) - set(stable)), dtype=int)
    # solve hitting probability problem (P-I)x = -b
    A = P[origin, :][:, origin] - np.eye((len(origin)))
    b = np.sum(-P[origin, :][:, target], axis=1)
    x = np.linalg.solve(A, b)
    # fill up full solution with 0's for stable states and 1's for target
    xfull = np.ones((n))
    xfull[origin] = x
    xfull[target] = 1
    xfull[stable] = 0

    return xfull