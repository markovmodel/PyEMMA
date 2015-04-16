
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

r"""Unit test, dense implementation of hitting probabilities

.. moduleauthor:: F.Noe <frank DOT noe AT fu-berlin DOT de>

"""
import unittest

import numpy as np
from pyemma.util.numeric import assert_allclose

from hitting_probability import hitting_probability


class TestHitting(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_hitting1(self):
        P = np.array([[0., 1., 0.],
                      [0., 1., 0.],
                      [0., 0., 1.]])
        sol = np.array([1, 0, 0])
        assert_allclose(hitting_probability(P, 1), sol)
        assert_allclose(hitting_probability(P, [1, 2]), sol)

    def test_hitting2(self):
        P = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.1, 0.8, 0.1, 0.0],
                      [0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.2, 0.8]])
        sol = np.array([0., 0.5, 1., 1.])
        assert_allclose(hitting_probability(P, [2, 3]), sol)

    def test_hitting3(self):
        P = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                      [0.1, 0.9, 0.0, 0.0, 0.0],
                      [0.0, 0.1, 0.4, 0.5, 0.0],
                      [0.0, 0.0, 0.0, 0.8, 0.2],
                      [0.0, 0.0, 0.0, 0.2, 0.8]])
        sol = np.array([0.0, 0.0, 8.33333333e-01, 1.0, 1.0])
        assert_allclose(hitting_probability(P, 3), sol)
        assert_allclose(hitting_probability(P, [3, 4]), sol)


if __name__ == "__main__":
    unittest.main()