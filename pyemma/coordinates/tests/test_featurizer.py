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

asn_leu_pdb = """
ATOM    559  N   ASN A  69      19.168  -0.936 -10.274  1.00 27.50           N  
ATOM    560  CA  ASN A  69      20.356  -0.049 -10.419  1.00 25.52           C  
ATOM    561  C   ASN A  69      21.572  -0.418  -9.653  1.00 24.26           C  
ATOM    562  O   ASN A  69      22.687  -0.336 -10.171  1.00 24.33           O  
ATOM    563  CB  ASN A  69      19.965   1.410 -10.149  1.00 26.49           C  
ATOM    564  CG  ASN A  69      18.932   1.881 -11.124  1.00 26.35           C  
ATOM    565  OD1 ASN A  69      18.835   1.322 -12.224  1.00 26.77           O  
ATOM    566  ND2 ASN A  69      18.131   2.864 -10.745  1.00 24.85           N  
ATOM    567  N   LEU A  70      21.419  -0.824  -8.404  1.00 23.02           N  
ATOM    568  CA  LEU A  70      22.592  -1.275  -7.656  1.00 23.37           C  
ATOM    569  C   LEU A  70      23.391  -2.325  -8.448  1.00 25.78           C  
ATOM    570  O   LEU A  70      24.647  -2.315  -8.430  1.00 25.47           O  
ATOM    571  CB  LEU A  70      22.202  -1.897  -6.306  1.00 22.17           C  
ATOM    572  CG  LEU A  70      23.335  -2.560  -5.519  1.00 22.49           C  
ATOM    573  CD1 LEU A  70      24.578  -1.665  -5.335  1.00 22.56           C  
ATOM    574  CD2 LEU A  70      22.853  -3.108  -4.147  1.00 24.47           C

""" *2 ### asn-leu-asn-leu


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
        cls.asn_leu_pdbfile = tempfile.mkstemp(suffix=".pdb")[1]
        with open(cls.asn_leu_pdbfile, 'w') as fh:
            fh.write(asn_leu_pdb)

        cls.asn_leu_traj = tempfile.mkstemp(suffix='.xtc')[1]

        # create traj for asn_leu
        n_frames = 4001
        traj = mdtraj.load(cls.asn_leu_pdbfile)
        ref = traj.xyz
        new_xyz = np.empty((n_frames, ref.shape[1], 3))
        noise = np.random.random(new_xyz.shape)
        new_xyz[:, :,: ] = noise + ref
        traj.xyz=new_xyz
        traj.time=np.arange(n_frames)
        traj.save(cls.asn_leu_traj)

        super(TestFeaturizer, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls.asn_leu_pdbfile)
        except EnvironmentError:
            pass
        super(TestFeaturizer, cls).tearDownClass()

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
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())


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

        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_dihedrals(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())

    def test_dihedrals_deg(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel, deg=True)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())

    def test_dihedrials_cossin(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel, cossin=True)
        assert(self.feat.dimension() == 2 * sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrals(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_backbone_torsions()

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.map(traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrals_deg(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_backbone_torsions(deg=True)

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.map(traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrals_cossin(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_backbone_torsions(cossin=True)

        traj = mdtraj.load(self.asn_leu_traj, top=self.asn_leu_pdbfile)
        Y = self.feat.map(traj)
        self.assertEqual(Y.shape, (len(traj), 3*4)) # (3 phi + 3 psi)*2 [cos, sin]
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        assert "COS" in desc[0]
        assert "SIN" in desc[1]
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrials_chi(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_chi1_torsions()

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.map(traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrials_chi_cossin(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_chi1_torsions(cossin=True)

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.map(traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        assert "COS" in desc[0]
        assert "SIN" in desc[1]
        self.assertEqual(len(desc), self.feat.dimension())

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
