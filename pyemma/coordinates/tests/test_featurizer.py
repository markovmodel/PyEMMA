# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import unittest
import numpy as np

import pyemma

import os
import mdtraj

from itertools import combinations, product

from pyemma.coordinates.data.featurization.featurizer import MDFeaturizer, CustomFeature
from pyemma.coordinates.data.featurization.util import _parse_pairwise_input, _describe_atom

from pyemma.coordinates.data.featurization.util import _atoms_in_residues
import pkg_resources

path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
xtcfile = os.path.join(path, 'bpti_mini.xtc')
pdbfile = os.path.join(path, 'bpti_ca.pdb')
pdbfile_ops_aa = os.path.join(path, 'opsin_aa_1_frame.pdb.gz')
pdbfile_ops_Ca = os.path.join(path, 'opsin_Ca_1_frame.pdb.gz')

asn_leu_pdb = """
ATOM    548  N   ARG A  68       5.907  -1.379  53.221  1.00 39.48           N  
ATOM    549  CA  ARG A  68       6.781  -2.058  54.196  1.00 40.75           C  
ATOM    550  C   ARG A  68       7.205  -3.453  53.719  1.00 40.21           C  
ATOM    551  O   ARG A  68       8.381  -3.819  53.821  1.00 37.97           O  
ATOM    552  CB  ARG A  68       6.101  -2.190  55.568  1.00 42.54           C  
ATOM    553  CG  ARG A  68       5.835  -0.874  56.293  1.00 44.54           C  
ATOM    554  CD  ARG A  68       5.539  -1.081  57.777  1.00 45.95           C  
ATOM    555  NE  ARG A  68       4.549  -2.141  58.029  1.00 47.49           N  
ATOM    556  CZ  ARG A  68       3.238  -1.977  58.259  1.00 48.16           C  
ATOM    557  NH1 ARG A  68       2.664  -0.774  58.288  1.00 48.92           N  
ATOM    558  NH2 ARG A  68       2.478  -3.050  58.470  1.00 48.77           N  
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
ATOM    659  N   ASN A  71      19.168  -0.936 -10.274  1.00 27.50           N  
ATOM    660  CA  ASN A  71      20.356  -0.049 -10.419  1.00 25.52           C  
ATOM    661  C   ASN A  71      21.572  -0.418  -9.653  1.00 24.26           C  
ATOM    662  O   ASN A  71      22.687  -0.336 -10.171  1.00 24.33           O  
ATOM    663  CB  ASN A  71      19.965   1.410 -10.149  1.00 26.49           C  
ATOM    664  CG  ASN A  71      18.932   1.881 -11.124  1.00 26.35           C  
ATOM    665  OD1 ASN A  71      18.835   1.322 -12.224  1.00 26.77           O  
ATOM    666  ND2 ASN A  71      18.131   2.864 -10.745  1.00 24.85           N  
ATOM    667  N   LEU A  72      21.419  -0.824  -8.404  1.00 23.02           N  
ATOM    668  CA  LEU A  72      22.592  -1.275  -7.656  1.00 23.37           C  
ATOM    669  C   LEU A  72      23.391  -2.325  -8.448  1.00 25.78           C  
ATOM    670  O   LEU A  72      24.647  -2.315  -8.430  1.00 25.47           O  
ATOM    671  CB  LEU A  72      22.202  -1.897  -6.306  1.00 22.17           C  
ATOM    672  CG  LEU A  72      23.335  -2.560  -5.519  1.00 22.49           C  
ATOM    673  CD1 LEU A  72      24.578  -1.665  -5.335  1.00 22.56           C  
ATOM    674  CD2 LEU A  72      22.853  -3.108  -4.147  1.00 24.47           C
"""  ### asn-leu-asn-leu

bogus_geom_pdbfile = """
ATOM    000  MW  ACE A  00      0.0000   0.000  0.0000  1.00 0.000           X
ATOM    001  CA  ASN A  01      1.0000   0.000  0.0000  1.00 0.000           C
ATOM    002  MW  ACE A  02      2.0000   0.000  0.0000  1.00 0.000           X
ATOM    003  CA  ASN A  03      3.0000   0.000  0.0000  1.00 0.000           C
ATOM    004  MW  ACE B  04      4.0000   0.000  0.0000  1.00 0.000           X
ATOM    005  CA  ASN B  05      5.0000   0.000  0.0000  1.00 0.000           C
ATOM    006  MW  ACE B  06      6.0000   0.000  0.0000  1.00 0.000           X
ATOM    007  CA  ASN B  07      7.0000   0.000  0.0000  1.00 0.000           C
"""


def verbose_assertion_minrmsd(ref_Y, test_Y, test_obj):
    for jj in np.arange(test_Y.shape[1]):
        ii = np.argmax(np.abs(ref_Y - test_Y[:, jj]))
        assert np.allclose(ref_Y, test_Y[:, jj], atol=test_obj.atol), \
            'Largest discrepancy between reference (ref_frame %u)' \
            ' and test: %8.2e, for the pair %f, %f at frame %u' % \
            (test_obj.ref_frame,
             (ref_Y - test_Y[:, jj])[ii],
             ref_Y[ii], test_Y[ii, jj], ii)


def check_serialized_equal(self):
    def feat_equal(a, b):
        assert isinstance(a, MDFeaturizer)
        assert isinstance(b, MDFeaturizer)
        self.assertEqual(a.dimension(), b.dimension())
        self.assertListEqual(a.describe(), b.describe())
        self.assertEqual(a.topology, b.topology)
        for f1, f2 in zip(a.active_features, b.active_features):
            if isinstance(f1, CustomFeature) or isinstance(f2, CustomFeature):
                # CustomFeatures are not equal after restoration, because we refuse to pickle functions (contexts).
                continue
            self.assertEqual(f1, f2, msg='%s != %s' %(f1,f2))
    feat = self.feat

    from pyemma.util.contexts import named_temporary_file
    with named_temporary_file() as buff:
        feat.save(buff)
        restored = pyemma.load(buff)
    feat_equal(restored, feat)


class TestFeaturizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.maxDiff = None
        import tempfile
        cls.asn_leu_pdbfile = tempfile.mkstemp(suffix=".pdb")[1]
        with open(cls.asn_leu_pdbfile, 'w') as fh:
            fh.write(asn_leu_pdb)

        cls.asn_leu_traj = tempfile.mktemp(suffix='.xtc')

        cls.bogus_geom_pdbfile = tempfile.mkstemp(suffix=".pdb")[1]
        with open(cls.bogus_geom_pdbfile, 'w') as fh:
            fh.write(bogus_geom_pdbfile)

        # create traj for asn_leu
        n_frames = 4001
        traj = mdtraj.load(cls.asn_leu_pdbfile)
        ref = traj.xyz
        new_xyz = np.empty((n_frames, ref.shape[1], 3))
        noise = np.random.random(new_xyz.shape)
        new_xyz[:, :, :] = noise + ref
        traj.xyz = new_xyz
        traj.time = np.arange(n_frames)
        traj.save(cls.asn_leu_traj)

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls.asn_leu_pdbfile)
        except EnvironmentError:
            pass

        try:
            os.unlink(cls.bogus_geom_pdbfile)
        except EnvironmentError:
            pass

    def setUp(self):
        self.pdbfile = pdbfile
        self.traj = mdtraj.load(xtcfile, top=self.pdbfile)
        self.feat = MDFeaturizer(self.pdbfile)
        self.atol = 1e-5
        self.ref_frame = 0
        self.atom_indices = np.arange(0, self.traj.n_atoms / 2)

    def tearDown(self):
        """
        before we destroy the featurizer created in each test, we dump it via
        serialization and restore it to check for equality.
        """
        check_serialized_equal(self)

    def test_select_backbone(self):
        inds = self.feat.select_Backbone()

    def test_select_non_symmetry_heavy_atoms(self):
        try:
            inds = self.feat.select_Heavy(exclude_symmetry_related=True)
        except RuntimeError as e:
            if "recursion depth" in e.args:
                import sys
                raise Exception("recursion limit reached. Interpreter limit: {}".format(sys.getrecursionlimit()))

    def test_select_all(self):
        self.feat.add_all()
        assert (self.feat.dimension() == self.traj.n_atoms * 3)
        refmap = np.reshape(self.traj.xyz, (len(self.traj), self.traj.n_atoms * 3))
        assert (np.all(refmap == self.feat.transform(self.traj)))

    def test_select_all_reference(self):
        self.feat.add_all(reference=self.traj[0])
        assert (self.feat.dimension() == self.traj.n_atoms * 3)
        aligned = self.traj.slice(slice(0, None), copy=True).superpose(self.traj[0])
        refmap = np.reshape(aligned.xyz, (len(self.traj), self.traj.n_atoms * 3))
        assert (np.all(refmap == self.feat.transform(self.traj)))

    def test_select(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        self.feat.add_selection(sel)
        assert (self.feat.dimension() == sel.shape[0] * 3)
        refmap = np.reshape(self.traj.xyz[:, sel, :], (len(self.traj), sel.shape[0] * 3))
        assert (np.all(refmap == self.feat.transform(self.traj)))

    def test_select_reference(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        self.feat.add_selection(sel, reference=self.traj[0])
        assert (self.feat.dimension() == sel.shape[0] * 3)
        aligned = self.traj.slice(slice(0, None), copy=True).superpose(self.traj[0])
        refmap = np.reshape(aligned.xyz[:, sel, :], (len(self.traj), sel.shape[0] * 3))
        np.testing.assert_equal(refmap, self.feat.transform(self.traj))

    def test_distances(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel, excluded_neighbors=2)
        assert (pairs.shape == pairs_expected.shape)
        assert (np.all(pairs == pairs_expected))
        self.feat.add_distances(pairs, periodic=False)  # unperiodic distances such that we can compare
        assert (self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert (np.allclose(D, self.feat.transform(self.traj)))

    def test_inverse_distances(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel, excluded_neighbors=2)
        assert (pairs.shape == pairs_expected.shape)
        assert (np.all(pairs == pairs_expected))
        self.feat.add_inverse_distances(pairs, periodic=False)  # unperiodic distances such that we can compare
        assert (self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        Dinv = 1.0 / np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert (np.allclose(Dinv, self.feat.transform(self.traj)))

    def test_ca_distances(self):
        sel = self.feat.select_Ca()
        assert (np.all(sel == list(range(self.traj.n_atoms))))  # should be all for this Ca-traj
        pairs = self.feat.pairs(sel, excluded_neighbors=0)
        self.feat.add_distances_ca(periodic=False,
                                   excluded_neighbors=0)  # unperiodic distances such that we can compare
        assert (self.feat.dimension() == pairs.shape[0])
        X = self.traj.xyz[:, pairs[:, 0], :]
        Y = self.traj.xyz[:, pairs[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert (np.allclose(D, self.feat.transform(self.traj)))

    def test_ca_distances_with_all_atom_geometries(self):
        feat = MDFeaturizer(pdbfile_ops_aa)
        feat.add_distances_ca(excluded_neighbors=0)
        D_aa = feat.transform(mdtraj.load(pdbfile_ops_aa))

        # Create a reference
        feat_just_ca = MDFeaturizer(pdbfile_ops_Ca)
        feat_just_ca.add_distances(np.arange(feat_just_ca.topology.n_atoms))
        D_ca = feat_just_ca.transform(mdtraj.load(pdbfile_ops_Ca))
        assert (np.allclose(D_aa, D_ca))

    def test_ca_distances_with_all_atom_geometries_and_exclusions(self):
        feat = MDFeaturizer(pdbfile_ops_aa)
        feat.add_distances_ca(excluded_neighbors=2)
        D_aa = feat.transform(mdtraj.load(pdbfile_ops_aa))

        # Create a reference
        feat_just_ca = MDFeaturizer(pdbfile_ops_Ca)
        ca_pairs = feat.pairs(feat_just_ca.select_Ca(), excluded_neighbors=2)
        feat_just_ca.add_distances(ca_pairs)
        D_ca = feat_just_ca.transform(mdtraj.load(pdbfile_ops_Ca))
        assert (np.allclose(D_aa, D_ca))

    def test_ca_distances_with_residues_not_containing_cas_no_exclusions(self):
        # Load test geom
        geom = mdtraj.load(self.pdbfile)
        # No exclusions
        feat_EN0 = MDFeaturizer(self.bogus_geom_pdbfile)
        feat_EN0.add_distances_ca(excluded_neighbors=0)
        ENO_pairs = [[1, 3], [1, 5], [1, 7],
                     [3, 5], [3, 7],
                     [5, 7]
                     ]

        # Check indices
        assert (np.allclose(ENO_pairs, feat_EN0.active_features[0].distance_indexes))
        # Check distances
        D = mdtraj.compute_distances(geom, ENO_pairs)
        assert (np.allclose(D, feat_EN0.transform(geom)))

        # excluded_neighbors=1 ## will yield the same as before, because the first neighbor
        # doesn't conting CA's anyway
        feat_EN1 = MDFeaturizer(self.bogus_geom_pdbfile)
        feat_EN1.add_distances_ca(excluded_neighbors=1)
        EN1_pairs = [[1, 3], [1, 5], [1, 7],
                     [3, 5], [3, 7],
                     [5, 7]
                     ]
        assert (np.allclose(EN1_pairs, feat_EN1.active_features[0].distance_indexes))
        D = mdtraj.compute_distances(geom, EN1_pairs)
        assert (np.allclose(D, feat_EN1.transform(geom)))

    def test_ca_distances_with_residues_not_containing_cas_with_exclusions(self):
        # Load test geom
        geom = mdtraj.load(self.pdbfile)
        # No exclusions
        feat_EN2 = MDFeaturizer(self.bogus_geom_pdbfile)
        feat_EN2.add_distances_ca(excluded_neighbors=2)
        EN2_pairs = [[1, 5], [1, 7],
                     [3, 7],
                     ]

        # Check indices
        assert (np.allclose(EN2_pairs, feat_EN2.active_features[0].distance_indexes))
        # Check distances
        D = mdtraj.compute_distances(geom, EN2_pairs)
        assert (np.allclose(D, feat_EN2.transform(geom)))

        # excluded_neighbors=1 ## will yield the same as before, because the first neighbor
        # doesn't conting CA's anyway
        feat_EN1 = MDFeaturizer(self.bogus_geom_pdbfile)
        feat_EN1.add_distances_ca(excluded_neighbors=1)
        EN1_pairs = [[1, 3], [1, 5], [1, 7],
                     [3, 5], [3, 7],
                     [5, 7]
                     ]
        assert (np.allclose(EN1_pairs, feat_EN1.active_features[0].distance_indexes))
        D = mdtraj.compute_distances(geom, EN1_pairs)
        assert (np.allclose(D, feat_EN1.transform(geom)))

    def test_contacts(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel, excluded_neighbors=2)
        assert (pairs.shape == pairs_expected.shape)
        assert (np.all(pairs == pairs_expected))
        self.feat.add_contacts(pairs, threshold=0.5, periodic=False)  # unperiodic distances such that we can compare
        assert (self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        C = np.zeros(D.shape)
        I = np.argwhere(D <= 0.5)
        C[I[:, 0], I[:, 1]] = 1.0
        assert (np.allclose(C, self.feat.transform(self.traj)))

    def test_contacts_count_contacts(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel, excluded_neighbors=2)
        assert (pairs.shape == pairs_expected.shape)
        assert (np.all(pairs == pairs_expected))
        self.feat.add_contacts(pairs, threshold=0.5, periodic=False,
                               count_contacts=True)  # unperiodic distances such that we can compare
        # The dimensionality of the feature is now one
        assert (self.feat.dimension() == 1)
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        C = np.zeros(D.shape)
        I = np.argwhere(D <= 0.5)
        C[I[:, 0], I[:, 1]] = 1.0
        # Count the contacts
        C = C.sum(1, keepdims=True)
        assert (np.allclose(C, self.feat.transform(self.traj)))

    def test_angles(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel)
        assert (self.feat.dimension() == sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert (np.alltrue(Y >= -np.pi))
        assert (np.alltrue(Y <= np.pi))
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())

    def test_angles_deg(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel, deg=True)
        assert (self.feat.dimension() == sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert (np.alltrue(Y >= -180.0))
        assert (np.alltrue(Y <= 180.0))

    def test_angles_cossin(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel, cossin=True)
        assert (self.feat.dimension() == 2 * sel.shape[0])
        Y = self.feat.transform(self.traj)
        self.assertEqual(Y.shape, (self.traj.n_frames, 2 * sel.shape[0]))
        assert (np.alltrue(Y >= -np.pi))
        assert (np.alltrue(Y <= np.pi))

        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_dihedrals(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel)
        assert (self.feat.dimension() == sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert (np.alltrue(Y >= -np.pi))
        assert (np.alltrue(Y <= np.pi))
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())

    def test_dihedrals_deg(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel, deg=True)
        assert (self.feat.dimension() == sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert (np.alltrue(Y >= -180.0))
        assert (np.alltrue(Y <= 180.0))
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())

    def test_dihedrials_cossin(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel, cossin=True)
        assert (self.feat.dimension() == 2 * sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert (np.alltrue(Y >= -np.pi))
        assert (np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrals(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_backbone_torsions()

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
        assert (np.alltrue(Y >= -np.pi))
        assert (np.alltrue(Y <= np.pi))

        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

        # test ordering of indices
        backbone_feature = self.feat.active_features[0]
        angle_indices = backbone_feature.angle_indexes
        np.testing.assert_equal(angle_indices[0], backbone_feature._phi_inds[0])
        np.testing.assert_equal(angle_indices[1], backbone_feature._psi_inds[0])
        np.testing.assert_equal(angle_indices[2], backbone_feature._phi_inds[1])
        np.testing.assert_equal(angle_indices[3], backbone_feature._psi_inds[1])

    def test_backbone_dihedrals_deg(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_backbone_torsions(deg=True)

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
        assert (np.alltrue(Y >= -180.0))
        assert (np.alltrue(Y <= 180.0))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrals_cossin(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_backbone_torsions(cossin=True)

        traj = mdtraj.load(self.asn_leu_traj, top=self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
        self.assertEqual(Y.shape, (len(traj), 2 * 8))  # (4 phi + 4 psi)*2 [cos, sin]
        assert (np.alltrue(Y >= -np.pi))
        assert (np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension(), msg=desc)
        self.assertIn("COS", desc[0])
        self.assertIn("SIN", desc[1])

    def test_backbone_dihedrials_chi1(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_sidechain_torsions(which='chi1')

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
        assert (np.alltrue(Y >= -np.pi))
        assert (np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrials_chi1_cossin(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_sidechain_torsions(cossin=True, which='chi1')

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
        assert (np.alltrue(Y >= -np.pi))
        assert (np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        assert "COS" in desc[0]
        assert "SIN" in desc[1]
        self.assertEqual(len(desc), self.feat.dimension())

    def test_all_dihedrals(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        # TODO: add chi5 when mdtraj-2.0 is released.
        self.feat.add_sidechain_torsions(which=['chi1', 'chi2', 'chi3', 'chi4'])
        assert self.feat.dimension() == 4 * 3  # 5 residues, chi1, chi2 (for 2*[asn, leu]), chi1-5 for arg

    def test_all_dihedrals_cossin(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        # TODO: add chi5 when mdtraj-2.0 is released.
        self.feat.add_sidechain_torsions(cossin=True, which=['chi1', 'chi2', 'chi3', 'chi4'])
        assert self.feat.dimension() == 2 * (4 * 3)
        desc = self.feat.describe()
        assert 'COS' in desc[0]
        assert 'SIN' in desc[1]

    def test_sidechain_torsions_which(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_sidechain_torsions(which='chi2')
        assert self.feat.dimension() == 5
        desc = self.feat.describe()
        assert all('CHI2' in d for d in desc)
        assert len(desc) == 5

    def test_sidechain_torsions_which2(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_sidechain_torsions(which=['chi1', 'chi3'])
        assert self.feat.dimension() == 6
        desc = self.feat.describe()
        assert len(desc) == 6

    def test_sidechain_torsions_selstr(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_sidechain_torsions(selstr='resid == 0', which=['chi1'])
        assert self.feat.dimension() == 1
        assert all('CHI1' in d for d in self.feat.describe())

    def test_sidechain_torsions_selstr_cos_which(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_sidechain_torsions(selstr='resid == 0', cossin=True, which=['chi1', 'chi2'])
        assert self.feat.dimension() == 4
        desc = self.feat.describe()
        assert any(['COS(CHI1' in d for d in desc])
        assert any(['SIN(CHI1' in d for d in desc])
        assert any(['COS(CHI2' in d for d in desc])
        assert any(['SIN(CHI2' in d for d in desc])

    def test_sidechain_torsions_invalid_which(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        with self.assertRaises(ValueError):
            self.feat.add_sidechain_torsions(selstr='resid == 0', which=['garbage'])

    def test_custom_feature(self):
        # TODO: test me
        pass

    def test_MinRmsd_ref_traj(self):
        # Test the Trajectory-input variant
        self.feat.add_minrmsd_to_ref(self.traj[self.ref_frame])

        test_Y = self.feat.transform(self.traj)
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame])
        verbose_assertion_minrmsd(ref_Y, test_Y, self)
        assert self.feat.dimension() == 1
        assert len(self.feat.describe()) == 1

    def test_MinRmsd_ref_file(self):
        # and the file-input variant
        self.feat.add_minrmsd_to_ref(xtcfile, ref_frame=self.ref_frame)
        test_Y = self.feat.transform(self.traj)
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame])
        verbose_assertion_minrmsd(ref_Y, test_Y, self)
        assert self.feat.dimension() == 1
        assert len(self.feat.describe()) == 1

    def test_MinRmsd_with_atom_indices(self):
        # Test the Trajectory-input variant
        self.feat.add_minrmsd_to_ref(self.traj[self.ref_frame], atom_indices=self.atom_indices)
        test_Y = self.feat.transform(self.traj)
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame], atom_indices=self.atom_indices)
        verbose_assertion_minrmsd(ref_Y, test_Y, self)
        assert self.feat.dimension() == 1
        assert len(self.feat.describe()) == 1

    def test_MinRmsd_with_atom_indices_precentered(self):
        # Test the Trajectory-input variant
        self.feat.add_minrmsd_to_ref(self.traj[self.ref_frame], atom_indices=self.atom_indices, precentered=True)
        test_Y = self.feat.transform(self.traj)
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame], atom_indices=self.atom_indices, precentered=True)
        verbose_assertion_minrmsd(ref_Y, test_Y, self)
        assert self.feat.dimension() == 1
        assert len(self.feat.describe()) == 1

    def test_Residue_Mindist_Ca_all(self):
        n_ca = self.feat.topology.n_atoms
        self.feat.add_residue_mindist(scheme='ca')
        D = self.feat.transform(self.traj)
        Dref = mdtraj.compute_contacts(self.traj, scheme='ca')[0]
        assert np.allclose(D, Dref)
        assert len(self.feat.describe()) == self.feat.dimension()

    def test_Residue_Mindist_Ca_all_threshold(self):
        threshold = .7
        self.feat.add_residue_mindist(scheme='ca', threshold=threshold)
        D = self.feat.transform(self.traj)
        Dref = mdtraj.compute_contacts(self.traj, scheme='ca')[0]
        Dbinary = np.zeros_like(Dref)
        I = np.argwhere(Dref <= threshold)
        Dbinary[I[:, 0], I[:, 1]] = 1
        assert np.allclose(D, Dbinary)
        assert len(self.feat.describe()) == self.feat.dimension()

    def test_Residue_Mindist_threshold_count_contacts(self):
        # residue pairs:
        pairs = self.feat.pairs(list(range(self.feat.topology.n_residues)), excluded_neighbors=2)
        self.feat.add_residue_mindist(scheme='ca', threshold=0.5, periodic=False, count_contacts=True)

        # The dimensionality of the feature is now one
        assert (self.feat.dimension() == 1)

        # number of upper triangular matrix elements excl two off-diagonals
        D = np.zeros((self.traj.n_frames,
                      int((self.feat.topology.n_residues**2 - 5 * self.feat.topology.n_residues + 6)/2)))

        for n, (resid_a, resid_b) in enumerate(pairs):
            # Ca only example: resid = atomid
            X = self.traj.xyz[:, [resid_a], :]
            Y = self.traj.xyz[:, [resid_b], :]
            D[:, n] = np.sqrt(np.sum((X - Y) ** 2, axis=2)).min(axis=1)

        C = (D <= 0.5).sum(axis=1, keepdims=True)

        assert (np.allclose(C, self.feat.transform(self.traj)))

    def test_Residue_Mindist_nothreshold_count_contacts(self):
        # residue pairs:
        with self.assertRaises(ValueError):
            self.feat.add_residue_mindist(scheme='ca', periodic=False, count_contacts=True)

    def test_Residue_Mindist_Ca_array(self):
        contacts = np.array([[20, 10, ], [10, 0]])
        self.feat.add_residue_mindist(scheme='ca', residue_pairs=contacts)
        D = self.feat.transform(self.traj)
        Dref = mdtraj.compute_contacts(self.traj, scheme='ca', contacts=contacts)[0]
        assert np.allclose(D, Dref)
        assert len(self.feat.describe()) == self.feat.dimension()

    def test_Residue_Mindist_Ca_array_periodic(self):
        traj = mdtraj.load(pdbfile)
        # Atoms most far appart in Z
        atom_minz = traj.xyz.argmin(1).squeeze()[-1]
        atom_maxz = traj.xyz.argmax(1).squeeze()[-1]
        # Residues with the atoms most far appart in Z
        res_minz = traj.topology.atom(atom_minz).residue.index
        res_maxz = traj.topology.atom(atom_maxz).residue.index
        contacts = np.array([[res_minz, res_maxz]])
        # Tweak the trajectory so that a (bogus) PBC exists (otherwise traj._have_unitcell is False)
        traj.unitcell_angles = [90, 90, 90]
        traj.unitcell_lengths = [1, 1, 1]
        self.feat.add_residue_mindist(scheme='ca', residue_pairs=contacts, periodic=False)
        D = self.feat.transform(traj)
        Dperiodic_true = mdtraj.compute_contacts(traj, scheme='ca', contacts=contacts, periodic=True)[0]
        Dperiodic_false = mdtraj.compute_contacts(traj, scheme='ca', contacts=contacts, periodic=False)[0]
        # This asserts that the periodic option is having an effect at all
        assert not np.allclose(Dperiodic_false, Dperiodic_true, )
        # This asserts that the periodic option is being handled correctly by pyemma
        assert np.allclose(D, Dperiodic_false)
        assert len(self.feat.describe()) == self.feat.dimension()

    def test_Group_Mindist_One_Group(self):
        group0 = [0, 20, 30, 0]
        self.feat.add_group_mindist(group_definitions=[group0])  # Even with duplicates
        D = self.feat.transform(self.traj)
        dist_list = list(combinations(np.unique(group0), 2))
        Dref = mdtraj.compute_distances(self.traj, dist_list)
        assert np.allclose(D.squeeze(), Dref.min(1))
        assert len(self.feat.describe()) == self.feat.dimension()

    def test_Group_Mindist_All_Three_Groups(self):
        group0 = [0, 20, 30, 0]
        group1 = [1, 21, 31, 1]
        group2 = [2, 22, 32, 2]
        self.feat.add_group_mindist(group_definitions=[group0, group1, group2])
        D = self.feat.transform(self.traj)

        # Now the references, computed separately for each combination of groups
        dist_list_01 = np.array(list(product(np.unique(group0), np.unique(group1))))
        dist_list_02 = np.array(list(product(np.unique(group0), np.unique(group2))))
        dist_list_12 = np.array(list(product(np.unique(group1), np.unique(group2))))
        Dref_01 = mdtraj.compute_distances(self.traj, dist_list_01).min(1)
        Dref_02 = mdtraj.compute_distances(self.traj, dist_list_02).min(1)
        Dref_12 = mdtraj.compute_distances(self.traj, dist_list_12).min(1)
        Dref = np.vstack((Dref_01, Dref_02, Dref_12)).T

        assert np.allclose(D.squeeze(), Dref)
        assert len(self.feat.describe()) == self.feat.dimension()

    def test_Group_Mindist_All_Three_Groups_threshold(self):
        threshold = .7
        group0 = [0, 20, 30, 0]
        group1 = [1, 21, 31, 1]
        group2 = [2, 22, 32, 2]
        self.feat.add_group_mindist(group_definitions=[group0, group1, group2], threshold=threshold)
        D = self.feat.transform(self.traj)

        # Now the references, computed separately for each combination of groups
        dist_list_01 = np.array(list(product(np.unique(group0), np.unique(group1))))
        dist_list_02 = np.array(list(product(np.unique(group0), np.unique(group2))))
        dist_list_12 = np.array(list(product(np.unique(group1), np.unique(group2))))
        Dref_01 = mdtraj.compute_distances(self.traj, dist_list_01).min(1)
        Dref_02 = mdtraj.compute_distances(self.traj, dist_list_02).min(1)
        Dref_12 = mdtraj.compute_distances(self.traj, dist_list_12).min(1)
        Dref = np.vstack((Dref_01, Dref_02, Dref_12)).T

        Dbinary = np.zeros_like(Dref)
        I = np.argwhere(Dref <= threshold)
        Dbinary[I[:, 0], I[:, 1]] = 1

        assert np.allclose(D, Dbinary)
        assert len(self.feat.describe()) == self.feat.dimension()

    def test_Group_Mindist_All_Three_Groups_threshold_count_contacts(self):
        threshold = .7
        group0 = [0, 20, 30, 0]
        group1 = [1, 21, 31, 1]
        group2 = [2, 22, 32, 2]
        self.feat.add_group_mindist(group_definitions=[group0, group1, group2],
                                    threshold=threshold, count_contacts=True)
        D = self.feat.transform(self.traj)

        # Now the references, computed separately for each combination of groups
        dist_list_01 = np.array(list(product(np.unique(group0), np.unique(group1))))
        dist_list_02 = np.array(list(product(np.unique(group0), np.unique(group2))))
        dist_list_12 = np.array(list(product(np.unique(group1), np.unique(group2))))
        Dref_01 = mdtraj.compute_distances(self.traj, dist_list_01).min(1)
        Dref_02 = mdtraj.compute_distances(self.traj, dist_list_02).min(1)
        Dref_12 = mdtraj.compute_distances(self.traj, dist_list_12).min(1)
        Dref = np.vstack((Dref_01, Dref_02, Dref_12)).T

        Dbinary = np.zeros_like(Dref)
        I = np.argwhere(Dref <= threshold)
        Dbinary[I[:, 0], I[:, 1]] = 1
        Dbinary_summed = Dbinary.sum(axis=1, keepdims=True)

        assert np.allclose(D, Dbinary_summed)

    def test_Group_Mindist_All_Three_Groups_nothreshold_count_contacts(self):
        group0 = [0, 20, 30, 0]
        group1 = [1, 21, 31, 1]
        group2 = [2, 22, 32, 2]
        with self.assertRaises(ValueError):
            self.feat.add_group_mindist(group_definitions=[group0, group1, group2], count_contacts=True)

    def test_Group_Mindist_Some_Three_Groups(self):
        group0 = [0, 20, 30, 0]
        group1 = [1, 21, 31, 1]
        group2 = [2, 22, 32, 2]

        group_pairs = np.array([[0, 1],
                                [2, 2],
                                [0, 2]])

        self.feat.add_group_mindist(group_definitions=[group0, group1, group2], group_pairs=group_pairs)
        D = self.feat.transform(self.traj)

        # Now the references, computed separately for each combination of groups
        dist_list_01 = np.array(list(product(np.unique(group0), np.unique(group1))))
        dist_list_02 = np.array(list(product(np.unique(group0), np.unique(group2))))
        dist_list_22 = np.array(list(combinations(np.unique(group2), 2)))
        Dref_01 = mdtraj.compute_distances(self.traj, dist_list_01).min(1)
        Dref_02 = mdtraj.compute_distances(self.traj, dist_list_02).min(1)
        Dref_22 = mdtraj.compute_distances(self.traj, dist_list_22).min(1)
        Dref = np.vstack((Dref_01, Dref_22, Dref_02)).T

        assert np.allclose(D.squeeze(), Dref)
        assert len(self.feat.describe()) == self.feat.dimension()

    # TODO consider creating a COM's own class and not a method of TestFeaturizer
    def test_Group_COM_with_all_atom_geoms(self):

        traj = mdtraj.load(pdbfile_ops_aa)
        traj = traj.join(traj)
        traj._xyz[-1] = traj.xyz[0] + np.array([10, 10, 10])  # The second frame's COM is the first plus 10

        # Needed variables for the checks
        group_definitions = [[0, 1, 3],
                             [4, 5, 6]
                             ]
        group_trajs = [traj.atom_slice(ra) for ra in group_definitions]

        feat = MDFeaturizer(traj.topology)
        feat.add_group_COM(group_definitions)

        # "Normal" COM, i.e. weighted
        ref_COM_xyz = np.hstack([mdtraj.compute_center_of_mass(itraj) for itraj in group_trajs])
        test_COM_xyz = feat.transform(traj)
        assert np.allclose(test_COM_xyz, ref_COM_xyz)

        # Unweighted COM (=geometric center)
        feat = MDFeaturizer(traj.topology)
        feat.add_group_COM(group_definitions, mass_weighted=False)

        ref_COM_xyz = np.hstack([np.mean(itraj.xyz, axis=1) for itraj in group_trajs])
        test_COM_xyz = feat.transform(traj)
        assert np.allclose(test_COM_xyz, ref_COM_xyz)

        # Using a reference (the second frame and first frame should NOT have different COMs anymore)
        feat = MDFeaturizer(traj.topology)
        feat.add_group_COM(group_definitions, mass_weighted=False, ref_geom=traj[0])
        test_COM_xyz = feat.transform(traj)
        assert np.allclose(test_COM_xyz[0], test_COM_xyz[1])

        # Using a image_molecule=True (just that it calls mdtrajs own method properly, not the method itself)
        feat = MDFeaturizer(traj.topology)
        feat.add_group_COM(group_definitions, image_molecules=True)
        test_COM_xyz = feat.transform(traj)

    def test_Residue_COM_with_all_atom_geoms(self):
        # The hard tests are in test_Group_COM, which is the superclass of ResidueCOMFeature
        # Here only sub-class specific things are tested, like the "scheme"
        traj = mdtraj.load(pdbfile_ops_aa)
        traj = traj.join(traj)

        # Using schemes (just that it works, the schemes themselves are  tested in  TestAtomsInResidues
        for scheme in ['all', 'backbone', 'sidechain']:
            feat = MDFeaturizer(traj.topology)
            feat.add_residue_COM(np.arange(traj.top.n_residues), scheme=scheme)
            feat.transform(traj)

        # And a full test to be sure
        # Needed variables for the checks
        residue_atoms = [traj.topology.select('resid %u' % (ii))
                         for ii in range(traj.n_residues)]
        residue_trajs = [traj.atom_slice(ra) for ra in residue_atoms]
        ref_COM_xyz = np.hstack([mdtraj.compute_center_of_mass(itraj) for itraj in residue_trajs])
        feat = MDFeaturizer(traj.topology)
        feat.add_residue_COM(np.arange(traj.top.n_residues))
        feat.transform(traj)
        test_COM_xyz = feat.transform(traj)
        assert np.allclose(test_COM_xyz, ref_COM_xyz)


class TestAtomsInResidues(unittest.TestCase):
    def setUp(self):
        self.traj = mdtraj.load(pdbfile_ops_aa)
        # Have a feature to have a logger to test all code
        self.feat = MDFeaturizer(self.traj.topology)

    def testAtomsInResidues_All_Schemes_NoFallBack_NoSubset(self):
        ref_atoms_in_residues = [self.traj.topology.select('resid %u' % ii)
                                 for ii in range(self.traj.n_residues)]

        test_atoms_in_residues = _atoms_in_residues(self.traj.top,
                                                    np.arange(self.traj.n_residues),
                                                    fallback_to_full_residue=False,
                                                    MDlogger=self.feat.logger)

        for ii, (ra1, ra2) in enumerate(zip(ref_atoms_in_residues, test_atoms_in_residues)):
            assert np.allclose(ra1, ra2)

    def testAtomsInResidues_All_Schemes_NoFallBack(self):
        for scheme in ['all', 'backbone', 'sidechain']:

            ref_atoms_in_residues = [self.traj.topology.select('resid %u and %s' % (ii, scheme))
                                     for ii in range(self.traj.n_residues)]

            test_atoms_in_residues = _atoms_in_residues(self.traj.top,
                                                        np.arange(self.traj.n_residues),
                                                        subset_of_atom_idxs=self.traj.topology.select(scheme),
                                                        fallback_to_full_residue=False,
                                                        MDlogger=self.feat.logger)

            for ii, (ra1, ra2) in enumerate(zip(ref_atoms_in_residues, test_atoms_in_residues)):
                assert np.allclose(ra1, ra2)

    def testAtomsInResidues_All_Schemes_FallBack(self):
        for scheme in ['all', 'backbone', 'sidechain']:
            ref_atoms_in_residues = [self.traj.topology.select('resid %u and %s' % (ii, scheme))
                                     for ii in range(self.traj.n_residues)]

            test_atoms_in_residues = _atoms_in_residues(self.traj.top,
                                                        np.arange(self.traj.n_residues),
                                                        subset_of_atom_idxs=self.traj.topology.select(scheme),
                                                        fallback_to_full_residue=True,
                                                        MDlogger=self.feat.logger)

            for ii, (ra1, ra2) in enumerate(zip(ref_atoms_in_residues, test_atoms_in_residues)):
                if len(ra1) == 0:  # means there are no atoms for this scheme, so we re-select without it
                    ra1 = self.traj.topology.select('resid %u' % ii)
                assert np.allclose(ra1, ra2)


class TestFeaturizerNoDubs(unittest.TestCase):
    def tearDown(self):
        """
        before we destroy the featurizer created in each test, we dump it via
        serialization and restore it to check for equality.
        """
        check_serialized_equal(self)

    def testAddFeaturesWithDuplicates(self):
        """this tests adds multiple features twice (eg. same indices) and
        checks whether they are rejected or not"""
        self.feat = MDFeaturizer(pdbfile)
        expected_active = 1

        self.feat.add_angles([[0, 1, 2], [0, 3, 4]])
        self.feat.add_angles([[0, 1, 2], [0, 3, 4]])
        self.assertEqual(len(self.feat.active_features), expected_active)

        self.feat.add_contacts([[0, 1], [0, 3]])
        expected_active += 1
        self.assertEqual(len(self.feat.active_features), expected_active)
        self.feat.add_contacts([[0, 1], [0, 3]])
        self.assertEqual(len(self.feat.active_features), expected_active)

        # try to fool it with ca selection
        ca = self.feat.select_Ca()
        ca = self.feat.pairs(ca, excluded_neighbors=0)
        self.feat.add_distances(ca)
        expected_active += 1
        self.assertEqual(len(self.feat.active_features), expected_active)
        self.feat.add_distances_ca(excluded_neighbors=0)
        self.assertEqual(len(self.feat.active_features), expected_active)

        self.feat.add_inverse_distances([[0, 1], [0, 3]])
        expected_active += 1
        self.assertEqual(len(self.feat.active_features), expected_active)

        self.feat.add_distances([[0, 1], [0, 3]])
        expected_active += 1
        self.assertEqual(len(self.feat.active_features), expected_active)
        self.feat.add_distances([[0, 1], [0, 3]])
        self.assertEqual(len(self.feat.active_features), expected_active)

        def my_func(x):
            return x - 1

        def foo(x):
            return x - 1

        expected_active += 1
        my_feature = CustomFeature(my_func, dim=3)
        self.feat.add_custom_feature(my_feature)

        self.assertEqual(len(self.feat.active_features), expected_active)
        self.feat.add_custom_feature(my_feature)
        self.assertEqual(len(self.feat.active_features), expected_active)

        # since myfunc and foo are different functions, it should be added
        expected_active += 1
        foo_feat = CustomFeature(foo, dim=3)
        self.feat.add_custom_feature(foo_feat)

        self.assertEqual(len(self.feat.active_features), expected_active)

        expected_active += 1
        ref = mdtraj.load(xtcfile, top=pdbfile)
        self.feat.add_minrmsd_to_ref(ref)
        self.feat.add_minrmsd_to_ref(ref)
        self.assertEqual(len(self.feat.active_features), expected_active)

        expected_active += 1
        self.feat.add_residue_mindist()
        self.feat.add_residue_mindist()
        self.assertEqual(len(self.feat.active_features), expected_active)

        expected_active += 1
        self.feat.add_group_mindist([[0, 1], [0, 2]])
        self.feat.add_group_mindist([[0, 1], [0, 2]])
        self.assertEqual(len(self.feat.active_features), expected_active)

        expected_active += 1
        self.feat.add_residue_COM([10, 20])
        self.feat.add_residue_COM([10, 20])
        self.assertEqual(len(self.feat.active_features), expected_active)

    def testAddVerySimilarResidueCOMs(self):
        traj = mdtraj.load(pdbfile_ops_aa)
        traj = traj.join(traj)
        traj._xyz[-1] = traj.xyz[0] + np.array([10, 10, 10])  # The second frame's COM is the first plus 10

        self.feat = MDFeaturizer(traj.topology)
        self.feat.add_residue_COM([0, 1, 2])
        self.feat.add_residue_COM([0, 1])
        self.feat.add_residue_COM([0, 1, ], mass_weighted=False)
        self.feat.add_residue_COM([0, 1, ], mass_weighted=False, image_molecules=True, scheme='backbone')
        self.feat.add_residue_COM([0, 1, ], mass_weighted=False, image_molecules=True, scheme='backbone', ref_geom=traj[0])
        self.feat.add_residue_COM([0, 1, ], mass_weighted=False, image_molecules=True, scheme='backbone', ref_geom=traj[1])
        assert len(self.feat.active_features) == 6

    def test_labels(self):
        """ just checks for exceptions """
        self.feat = MDFeaturizer(pdbfile)
        self.feat.add_angles([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError) as cm:
            self.feat.add_backbone_torsions()
            assert 'emtpy indices' in cm.exception.message
        self.feat.add_contacts([[0, 1], [0, 3]])
        self.feat.add_distances([[0, 1], [0, 3]])
        self.feat.add_inverse_distances([[0, 1], [0, 3]])
        cs = CustomFeature(lambda x: x - 1, dim=3)
        self.feat.add_custom_feature(cs)
        self.feat.add_minrmsd_to_ref(pdbfile)
        self.feat.add_residue_mindist()
        self.feat.add_group_mindist([[0, 1], [0, 2]])
        self.feat.add_residue_COM([0, 1, 2])

        self.feat.describe()


class TestPairwiseInputParser(unittest.TestCase):

    def setUp(self):
        self.feat = MDFeaturizer(pdbfile)

    def test_trivial(self):
        dist_list = np.array([[0, 1],
                              [0, 2],
                              [0, 3]])

        assert np.allclose(dist_list, _parse_pairwise_input(dist_list, None, self.feat.logger))

    def test_one_unique(self):
        # As a list
        group1 = [0, 1, 2]
        dist_list = np.asarray(list(combinations(group1, 2)))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, None, self.feat.logger))

        # As an array
        group1 = np.array([0, 1, 2])
        dist_list = np.asarray(list(combinations(group1, 2)))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, None, self.feat.logger))

    def test_two_uniques(self):
        # As a list
        group1 = [0, 1, 2]
        group2 = [3, 4, 5]
        dist_list = np.asarray(list(product(group1, group2)))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, group2, self.feat.logger))

        # As an array
        group1 = np.array([0, 1, 2])
        group2 = np.array([3, 4, 5])
        dist_list = np.asarray(list(product(group1, group2)))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, group2, self.feat.logger))

    def test_two_redundants(self):
        group1 = np.array([0, 1, 2, 0])
        group2 = np.array([3, 4, 5, 4])
        dist_list = np.asarray(list(product(np.unique(group1),
                                            np.unique(group2)
                                            )))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, group2, self.feat.logger))

    def test_two_redundants_overlap(self):
        group1 = np.array([0, 1, 2, 0])
        group2 = np.array([3, 4, 5, 4, 0, 1])
        dist_list = np.asarray(list(product(np.unique(group1),
                                            np.unique(group2[:-2])
                                            )))
        assert np.allclose(dist_list, _parse_pairwise_input(group1, group2, self.feat.logger))


class TestUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tempfile
        cls.bogus_geom_pdbfile = tempfile.mkstemp(suffix=".pdb")[1]
        print(cls.bogus_geom_pdbfile)
        with open(cls.bogus_geom_pdbfile, 'w') as fh:
            fh.write(bogus_geom_pdbfile)
        super(TestUtils, cls).setUpClass()

    @classmethod
    def tearDownClass(cls):
        try:
            os.unlink(cls.bogus_geom_pdbfile)
        except EnvironmentError:
            pass

        super(TestUtils, cls).tearDownClass()

    @classmethod
    def setUp(self):
        self.traj = mdtraj.load(self.bogus_geom_pdbfile)

    def test_describe_atom(self):
        str1 = _describe_atom(self.traj.topology, 0)
        str2 = _describe_atom(self.traj.topology, self.traj.n_atoms - 1)
        assert len(str1.split()) >= 4
        assert len(str2.split()) >= 4
        assert str1.split()[-1] == '0'
        assert str2.split()[-1] == '1'


class TestStaticMethods(unittest.TestCase):

    def setUp(self):
        self.feat = MDFeaturizer(pdbfile)

    def test_pairs(self):
        n_at = 5
        pairs = self.feat.pairs(np.arange(n_at), excluded_neighbors=3)
        assert np.allclose(pairs, [0, 4])

        pairs = self.feat.pairs(np.arange(n_at), excluded_neighbors=2)
        assert np.allclose(pairs, [[0, 3], [0, 4],
                                   [1, 4]])

        pairs = self.feat.pairs(np.arange(n_at), excluded_neighbors=1)
        assert np.allclose(pairs, [[0, 2], [0, 3], [0, 4],
                                   [1, 3], [1, 4],
                                   [2, 4]])

        pairs = self.feat.pairs(np.arange(n_at), excluded_neighbors=0)
        assert np.allclose(pairs, [[0, 1], [0, 2], [0, 3], [0, 4],
                                   [1, 2], [1, 3], [1, 4],
                                   [2, 3], [2, 4],
                                   [3, 4]])


# Define some function that somehow mimics one would typically want to do,
# e.g. 1. call mdtraj,
#      2. perform some other operations on the result
#      3. return a numpy array
def some_call_to_mdtraj_some_operations_some_linalg(traj, pairs, means, U):
    D = mdtraj.compute_distances(traj, pairs)
    D_meanfree = D - means
    Y = (U.T.dot(D_meanfree.T)).T
    return Y.astype('float32')


class TestCustomFeature(unittest.TestCase):

    def setUp(self):
        self.feat = MDFeaturizer(pdbfile)
        self.traj = mdtraj.load(xtcfile, top=pdbfile)

        self.pairs = [[0, 1], [0, 2], [1, 2]]  # some distances
        self.means = [.5, .75, 1.0]  # bogus means
        self.U = np.array([[0, 1],
                           [1, 0],
                           [1, 1]])  # bogus transformation, projects from 3 distances to 2 components

    def test_some_feature(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1],
                                  self.pairs,
                                  self.means,
                                  self.U
                                  )

        Y_custom_feature = self.feat.transform(self.traj)
        # Directly call the function
        Y_function = some_call_to_mdtraj_some_operations_some_linalg(self.traj, self.pairs, self.means, self.U)
        assert np.allclose(Y_custom_feature, Y_function)

    def test_describe(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1],
                                  self.pairs,
                                  self.means,
                                  self.U
                                  )
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_describe_given(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1],
                                  self.pairs,
                                  self.means,
                                  self.U, description=['foo'] * self.U.shape[1]
                                  )
        desc = self.feat.describe()
        self.assertIn('foo', desc)
        self.assertEqual(len(desc), self.feat.dimension())

    def test_describe_given_str(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1],
                                  self.pairs,
                                  self.means,
                                  self.U, description='test')
        desc = self.feat.describe()
        self.assertIn('test', desc)
        self.assertEqual(len(desc), self.feat.dimension())

    def test_describe_given_wrong(self):
        """ either a list matching input dim, or 1 element iterable allowed"""
        with self.assertRaises(ValueError) as cm:
            self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1] + 1,
                                      self.pairs,
                                      self.means,
                                      self.U, description=['ff', 'ff'])

    def test_describe_1_element_expand(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1] + 1,
                                  self.pairs,
                                  self.means,
                                  self.U, description=['test'])
        desc = self.feat.describe()
        self.assertEqual(desc, ['test'] * 3)

    def test_dimensionality(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1],
                                  self.pairs,
                                  self.means,
                                  self.U
                                  )

        assert self.feat.dimension() == self.U.shape[1]

    def test_serializable(self):
        import tempfile
        f = tempfile.mktemp()
        try:
            self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1],
                                      self.pairs,
                                      self.means,
                                      self.U
                                      )
            self.feat.save(f)
            from pyemma import load
            restored = load(f)
            with self.assertRaises(NotImplementedError) as cw:
                restored.transform(self.traj)
            self.assertIn('re-add your custom feature', cw.exception.args[0])
        finally:
            import os
            os.unlink(f)

if __name__ == "__main__":
    unittest.main()
