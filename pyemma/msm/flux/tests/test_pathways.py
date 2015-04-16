
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

"""Unit test for the reaction pathway decomposition

.. moduleauthor:: B.Trendelkamp-Schroer <benjamin DOT trendelkamp-schroer AT fu-berlin DOT de>

"""

import warnings
import unittest
import numpy as np
from scipy.sparse import csr_matrix

from pyemma.util.numeric import assert_allclose
from pyemma.msm.flux import pathways


class TestPathways(unittest.TestCase):
    
    def setUp(self):
        """Small flux-network"""
        F = np.zeros((8, 8))
        F[0, 2] = 10.0
        F[2, 6] = 10.0    
        F[1, 3] = 100.0
        F[3, 4] = 30.0
        F[3, 5] = 70.0
        F[4, 6] = 5.0
        F[4, 7] = 25.0
        F[5, 6] = 30.0
        F[5, 7] = 40.0        
        """Reactant and product states"""
        A = [0, 1]
        B = [6, 7]
    
        self.F = F
        self.F_sparse = csr_matrix(F)
        self.A = A
        self.B = B
        self.paths = []
        self.capacities = []
        p1 = np.array([1, 3, 5, 7])
        c1 = 40.0
        self.paths.append(p1)
        self.capacities.append(c1)
        p2 = np.array([1, 3, 5, 6])
        c2 = 30.0
        self.paths.append(p2)
        self.capacities.append(c2)
        p3 = np.array([1, 3, 4, 7])
        c3 = 25.0
        self.paths.append(p3)
        self.capacities.append(c3)
        p4 = np.array([0, 2, 6])
        c4 = 10.0
        self.paths.append(p4)
        self.capacities.append(c4)
        p5 = np.array([1, 3, 4, 6])
        c5 = 5.0
        self.paths.append(p5)
        self.capacities.append(c5)

    def test_pathways_dense(self):
        paths, capacities = pathways(self.F, self.A, self.B)
        self.assertTrue(len(paths) == len(self.paths))
        self.assertTrue(len(capacities) == len(self.capacities))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

    def test_pathways_dense_incomplete(self):
        paths, capacities = pathways(self.F, self.A, self.B, fraction=0.5)
        self.assertTrue(len(paths) == len(self.paths[0:2]))
        self.assertTrue(len(capacities) == len(self.capacities[0:2]))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            paths, capacities = pathways(self.F, self.A, self.B, fraction=1.0, maxiter=1)
            for i in range(len(paths)):
                assert_allclose(paths[i], self.paths[i])
                assert_allclose(capacities[i], self.capacities[i])            
            assert issubclass(w[-1].category, RuntimeWarning)        

    def test_pathways_sparse(self):
        paths, capacities = pathways(self.F_sparse, self.A, self.B)
        self.assertTrue(len(paths) == len(self.paths))
        self.assertTrue(len(capacities) == len(self.capacities))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

    def test_pathways_sparse_incomplete(self):
        paths, capacities = pathways(self.F_sparse, self.A, self.B, fraction=0.5)
        self.assertTrue(len(paths) == len(self.paths[0:2]))
        self.assertTrue(len(capacities) == len(self.capacities[0:2]))

        for i in range(len(paths)):
            assert_allclose(paths[i], self.paths[i])
            assert_allclose(capacities[i], self.capacities[i])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            paths, capacities = pathways(self.F, self.A, self.B, fraction=1.0, maxiter=1)      
            for i in range(len(paths)):
                assert_allclose(paths[i], self.paths[i])
                assert_allclose(capacities[i], self.capacities[i])            
            assert issubclass(w[-1].category, RuntimeWarning)
        
if __name__ == "__main__":
    unittest.main()