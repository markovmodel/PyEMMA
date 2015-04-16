
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

r"""Unit tests for the committor API-function

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import unittest
import numpy as np
from pyemma.util.numeric import assert_allclose

from pyemma.msm.analysis import committor

from birth_death_chain import BirthDeathChain


class TestCommittorDense(unittest.TestCase):
    def setUp(self):
        p = np.zeros(10)
        q = np.zeros(10)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[4] = 0.01
        q[6] = 0.1

        self.bdc = BirthDeathChain(q, p)

    def tearDown(self):
        pass

    def test_forward_comittor(self):
        P = self.bdc.transition_matrix()
        un = committor(P, [0, 1], [8, 9], forward=True)
        u = self.bdc.committor_forward(1, 8)
        assert_allclose(un, u)

    def test_backward_comittor(self):
        P = self.bdc.transition_matrix()
        un = committor(P, [0, 1], [8, 9], forward=False)
        u = self.bdc.committor_backward(1, 8)
        assert_allclose(un, u)
        

class TestCommittorSparse(unittest.TestCase):
    def setUp(self):
        p = np.zeros(100)
        q = np.zeros(100)
        p[0:-1] = 0.5
        q[1:] = 0.5
        p[49] = 0.01
        q[51] = 0.1

        self.bdc = BirthDeathChain(q, p)

    def tearDown(self):
        pass

    def test_forward_comittor(self):
        P = self.bdc.transition_matrix_sparse()
        un = committor(P, range(10), range(90, 100), forward=True)
        u = self.bdc.committor_forward(9, 90)
        assert_allclose(un, u)

    def test_backward_comittor(self):
        P = self.bdc.transition_matrix_sparse()
        un = committor(P, range(10), range(90, 100), forward=False)
        u = self.bdc.committor_backward(9, 90)
        assert_allclose(un, u)

if __name__ == "__main__":
    unittest.main()