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
import unittest
import numpy as np

import os
import mdtraj

from itertools import combinations, product

# from pyemma.coordinates.data import featurizer as ft
from pyemma.coordinates.data.featurizer import MDFeaturizer, CustomFeature, _parse_pairwise_input
# from pyemma.coordinates.tests.test_discretizer import create_water_topology_on_disc

path = os.path.join(os.path.split(__file__)[0], 'data')
xtcfile = os.path.join(path, 'bpti_mini.xtc')
pdbfile = os.path.join(path, 'bpti_ca.pdb')

ala_dipetide_pdb = \
""" REMARK  ACE
ATOM      1 1HH3 ACE     1       2.000   1.000  -0.000
ATOM      2  CH3 ACE     1       2.000   2.090   0.000
ATOM      3 2HH3 ACE     1       1.486   2.454   0.890
ATOM      4 3HH3 ACE     1       1.486   2.454  -0.890
ATOM      5  C   ACE     1       3.427   2.641  -0.000
ATOM      6  O   ACE     1       4.391   1.877  -0.000
ATOM      7  N   ALA     2       3.555   3.970  -0.000
ATOM      8  H   ALA     2       2.733   4.556  -0.000
ATOM      9  CA  ALA     2       4.853   4.614  -0.000
ATOM     10  HA  ALA     2       5.408   4.316   0.890
ATOM     11  CB  ALA     2       5.661   4.221  -1.232
ATOM     12 1HB  ALA     2       5.123   4.521  -2.131
ATOM     13 2HB  ALA     2       6.630   4.719  -1.206
ATOM     14 3HB  ALA     2       5.809   3.141  -1.241
ATOM     15  C   ALA     2       4.713   6.129   0.000
ATOM     16  O   ALA     2       3.601   6.653   0.000
ATOM     17  N   NME     3       5.846   6.835   0.000
ATOM     18  H   NME     3       6.737   6.359  -0.000
ATOM     19  CH3 NME     3       5.846   8.284   0.000
ATOM     20 1HH3 NME     3       4.819   8.648   0.000
ATOM     21 2HH3 NME     3       6.360   8.648   0.890
ATOM     22 3HH3 NME     3       6.360   8.648  -0.890
TER
END
"""


def verbose_assertion_minrmsd(ref_Y, test_Y, test_obj):
    for jj in np.arange(test_Y.shape[1]):
        ii = np.argmax(np.abs(ref_Y-test_Y[:,jj]))
        assert np.allclose(ref_Y, test_Y[:,jj], atol=test_obj.atol), \
            'Largest discrepancy between reference (ref_frame %u)' \
            ' and test: %8.2e, for the pair %f, %f at frame %u'%\
            (test_obj.ref_frame,
             (ref_Y-test_Y[:,jj])[ii],
             ref_Y[ii], test_Y[ii,jj], ii)


class TestFeaturizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.ala_pdb = tempfile.mkstemp(suffix=".pdb")[1]
        with open(cls.ala_pdb, 'w') as fh:
            fh.write(ala_dipetide_pdb)

        super(TestFeaturizer, cls).setUpClass()

    def setUp(self):
        self.pdbfile = pdbfile
        self.traj = mdtraj.load(xtcfile, top=self.pdbfile)
        self.feat = MDFeaturizer(self.pdbfile)
        self.atol = 1e-5
        self.ref_frame = 0
        self.atom_indices = np.arange(0, self.traj.n_atoms/2)

    def test_select_backbone(self):
        inds = self.feat.select_Backbone()

    def test_select_all(self):
        self.feat.add_all()
        assert (self.feat.dimension() == self.traj.n_atoms * 3)
        refmap = np.reshape(self.traj.xyz, (len(self.traj), self.traj.n_atoms * 3))
        assert (np.all(refmap == self.feat.map(self.traj)))

    def test_select(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        self.feat.add_selection(sel)
        assert (self.feat.dimension() == sel.shape[0] * 3)
        refmap = np.reshape(self.traj.xyz[:, sel, :], (len(self.traj), sel.shape[0] * 3))
        assert (np.all(refmap == self.feat.map(self.traj)))

    def test_distances(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_distances(pairs, periodic=False)  # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(D, self.feat.map(self.traj)))

    def test_inverse_distances(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_inverse_distances(pairs, periodic=False)  # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        Dinv = 1.0/np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(Dinv, self.feat.map(self.traj)))

    def test_ca_distances(self):
        sel = self.feat.select_Ca()
        assert(np.all(sel == range(self.traj.n_atoms)))  # should be all for this Ca-traj
        pairs = self.feat.pairs(sel)
        self.feat.add_distances_ca(periodic=False)  # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs.shape[0])
        X = self.traj.xyz[:, pairs[:, 0], :]
        Y = self.traj.xyz[:, pairs[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(D, self.feat.map(self.traj)))

    def test_contacts(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_contacts(pairs, threshold=0.5, periodic=False)  # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        C = np.zeros(D.shape)
        I = np.argwhere(D <= 0.5)
        C[I[:, 0], I[:, 1]] = 1.0
        assert(np.allclose(C, self.feat.map(self.traj)))

    def test_angles(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

    def test_angles_deg(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel, deg=True)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))

    def test_angles_cossin(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel, cossin=True)
        assert(self.feat.dimension() == 2 * sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

        self.feat.describe()

    def test_dihedrals(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

    def test_dihedrals_deg(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel, deg=True)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))

    def test_backbone_dihedrals(self):
        self.feat = MDFeaturizer(topfile=self.ala_pdb)
        self.feat.add_backbone_torsions()

        traj = mdtraj.load(self.ala_pdb)
        Y = self.feat.map(traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

        self.feat.describe()

    def test_backbone_dihedrals_deg(self):
        self.feat = MDFeaturizer(topfile=self.ala_pdb)
        self.feat.add_backbone_torsions(deg=True)

        traj = mdtraj.load(self.ala_pdb)
        Y = self.feat.map(traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))
        desc = self.feat.describe()

    def test_backbone_dihedrals_cossin(self):
        self.feat = MDFeaturizer(topfile=self.ala_pdb)
        self.feat.add_backbone_torsions(cossin=True)

        traj = mdtraj.load(self.ala_pdb)
        Y = self.feat.map(traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        assert "SIN" in desc[0]
        assert "COS" in desc[0]

    def test_custom_feature(self):
        # TODO: test me
        pass

    def test_MinRmsd(self):
        # Test the Trajectory-input variant
        self.feat.add_minrmsd_to_ref(self.traj[self.ref_frame])
        # and the file-input variant
        self.feat.add_minrmsd_to_ref(xtcfile, ref_frame=self.ref_frame)
        test_Y  = self.feat.map(self.traj).squeeze()
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame])
        verbose_assertion_minrmsd(ref_Y, test_Y, self)

    def test_MinRmsd_with_atom_indices(self):
        # Test the Trajectory-input variant
        self.feat.add_minrmsd_to_ref(self.traj[self.ref_frame], atom_indices=self.atom_indices)
        # and the file-input variant
        self.feat.add_minrmsd_to_ref(xtcfile, ref_frame=self.ref_frame, atom_indices=self.atom_indices)
        test_Y  = self.feat.map(self.traj).squeeze()
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame], atom_indices=self.atom_indices)
        verbose_assertion_minrmsd(ref_Y, test_Y, self)

    def test_MinRmsd_with_atom_indices_precentered(self):
        # Test the Trajectory-input variant
        self.feat.add_minrmsd_to_ref(self.traj[self.ref_frame], atom_indices=self.atom_indices, precentered=True)
        # and the file-input variant
        self.feat.add_minrmsd_to_ref(xtcfile, ref_frame=self.ref_frame, atom_indices=self.atom_indices, precentered=True)
        test_Y  = self.feat.map(self.traj).squeeze()
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame], atom_indices=self.atom_indices, precentered=True)
        verbose_assertion_minrmsd(ref_Y, test_Y, self)

class TestFeaturizerNoDubs(unittest.TestCase):

    def testAddFeaturesWithDuplicates(self):
        """this tests adds multiple features twice (eg. same indices) and
        checks whether they are rejected or not"""
        featurizer = MDFeaturizer(pdbfile)

        featurizer.add_angles([[0, 1, 2], [0, 3, 4]])
        featurizer.add_angles([[0, 1, 2], [0, 3, 4]])

        self.assertEqual(len(featurizer.active_features), 1)

        featurizer.add_backbone_torsions()

        self.assertEqual(len(featurizer.active_features), 2)
        featurizer.add_backbone_torsions()
        self.assertEqual(len(featurizer.active_features), 2)

        featurizer.add_contacts([[0, 1], [0, 3]])
        self.assertEqual(len(featurizer.active_features), 3)
        featurizer.add_contacts([[0, 1], [0, 3]])
        self.assertEqual(len(featurizer.active_features), 3)

        # try to fool it with ca selection
        ca = featurizer.select_Ca()
        ca = featurizer.pairs(ca)
        featurizer.add_distances(ca)
        self.assertEqual(len(featurizer.active_features), 4)
        featurizer.add_distances_ca()
        self.assertEqual(len(featurizer.active_features), 4)

        featurizer.add_inverse_distances([[0, 1], [0, 3]])
        self.assertEqual(len(featurizer.active_features), 5)

        featurizer.add_distances([[0, 1], [0, 3]])
        self.assertEqual(len(featurizer.active_features), 6)
        featurizer.add_distances([[0, 1], [0, 3]])
        self.assertEqual(len(featurizer.active_features), 6)

        def my_func(x):
            return x - 1

        def foo(x):
            return x - 1

        my_feature = CustomFeature(my_func)
        my_feature.dimension = 3
        featurizer.add_custom_feature(my_feature)

        self.assertEqual(len(featurizer.active_features), 7)
        featurizer.add_custom_feature(my_feature)
        self.assertEqual(len(featurizer.active_features), 7)
        # since myfunc and foo are different functions, it should be added
        foo_feat = CustomFeature(foo, dim=3)
        featurizer.add_custom_feature(foo_feat)
        self.assertEqual(len(featurizer.active_features), 8)

        ref = mdtraj.load(xtcfile, top=pdbfile)
        featurizer.add_minrmsd_to_ref(ref)
        featurizer.add_minrmsd_to_ref(ref)
        self.assertEquals(len(featurizer.active_features), 9)

        featurizer.add_minrmsd_to_ref(pdbfile)
        featurizer.add_minrmsd_to_ref(pdbfile)
        self.assertEquals(len(featurizer.active_features), 10)

    def test_labels(self):
        """ just checks for exceptions """
        featurizer = MDFeaturizer(pdbfile)
        featurizer.add_angles([[1, 2, 3], [4, 5, 6]])
        featurizer.add_backbone_torsions()
        featurizer.add_contacts([[0, 1], [0, 3]])
        featurizer.add_distances([[0, 1], [0, 3]])
        featurizer.add_inverse_distances([[0, 1], [0, 3]])
        cs = CustomFeature(lambda x: x - 1)
        cs.dimension = lambda: 3
        featurizer.add_custom_feature(cs)
        featurizer.add_minrmsd_to_ref(pdbfile)

        featurizer.describe()

class TestPairwiseInputParser(unittest.TestCase):

    def setUp(self):
        self.feat = MDFeaturizer(pdbfile)

    def test_trivial(self):
        dist_list = np.array([[0, 1],
                              [0, 2],
                              [0, 3]])

        assert np.allclose(dist_list, _parse_pairwise_input(dist_list, None, self.feat._logger))

    def test_one_unique(self):
        # As a list
        group1 = [0, 1, 2]
        dist_list = np.asarray(list(combinations(group1, 2)))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, None, self.feat._logger))

        # As an array
        group1 = np.array([0, 1, 2])
        dist_list = np.asarray(list(combinations(group1, 2)))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, None, self.feat._logger))

    def test_two_uniques(self):
        # As a list
        group1 = [0, 1, 2]
        group2 = [3, 4, 5]
        dist_list = np.asarray(list(product(group1, group2)))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, group2, self.feat._logger))

        # As an array
        group1 = np.array([0, 1, 2])
        group2 = np.array([3, 4, 5])
        dist_list = np.asarray(list(product(group1, group2)))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, group2, self.feat._logger))

    def test_two_redundants(self):
        group1 = np.array([0, 1, 2, 0])
        group2 = np.array([3, 4, 5, 4])
        dist_list = np.asarray(list(product(np.unique(group1),
                                            np.unique(group2)
                                            )))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, group2, self.feat._logger))

    def test_two_redundants_overlap(self):
        group1 = np.array([0, 1, 2, 0])
        group2 = np.array([3, 4, 5, 4, 0, 1])
        dist_list = np.asarray(list(product(np.unique(group1),
                                            np.unique(group2[:-2])
                                            )))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, group2, self.feat._logger))

if __name__ == "__main__":
    unittest.main()
