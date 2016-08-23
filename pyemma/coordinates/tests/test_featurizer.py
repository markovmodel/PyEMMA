
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


from __future__ import absolute_import
import unittest
import numpy as np

import os
import mdtraj

from itertools import combinations, product

# from pyemma.coordinates.data import featurizer as ft
from pyemma.coordinates.data.featurization.featurizer import MDFeaturizer, CustomFeature
from pyemma.coordinates.data.featurization.util import _parse_pairwise_input, _describe_atom
from six.moves import range
import pkg_resources

path = pkg_resources.resource_filename(__name__, 'data') + os.path.sep
xtcfile = os.path.join(path, 'bpti_mini.xtc')
pdbfile = os.path.join(path, 'bpti_ca.pdb')
pdbfile_ops_aa = os.path.join(path,'opsin_aa_1_frame.pdb.gz')
pdbfile_ops_Ca = os.path.join(path,'opsin_Ca_1_frame.pdb.gz')

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
        new_xyz[:, :,: ] = noise + ref
        traj.xyz=new_xyz
        traj.time=np.arange(n_frames)
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
        self.atom_indices = np.arange(0, self.traj.n_atoms/2)

    def test_select_backbone(self):
        inds = self.feat.select_Backbone()
    
    def test_select_non_symmetry_heavy_atoms(self):
        inds = self.feat.select_Heavy(exclude_symmetry_related=True)

    def test_select_all(self):
        self.feat.add_all()
        assert (self.feat.dimension() == self.traj.n_atoms * 3)
        refmap = np.reshape(self.traj.xyz, (len(self.traj), self.traj.n_atoms * 3))
        assert (np.all(refmap == self.feat.transform(self.traj)))

    def test_select(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        self.feat.add_selection(sel)
        assert (self.feat.dimension() == sel.shape[0] * 3)
        refmap = np.reshape(self.traj.xyz[:, sel, :], (len(self.traj), sel.shape[0] * 3))
        assert (np.all(refmap == self.feat.transform(self.traj)))

    def test_distances(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel, excluded_neighbors=2)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_distances(pairs, periodic=False)  # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(D, self.feat.transform(self.traj)))

    def test_inverse_distances(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel, excluded_neighbors=2)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_inverse_distances(pairs, periodic=False)  # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        Dinv = 1.0/np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(Dinv, self.feat.transform(self.traj)))

    def test_ca_distances(self):
        sel = self.feat.select_Ca()
        assert(np.all(sel == list(range(self.traj.n_atoms))))  # should be all for this Ca-traj
        pairs = self.feat.pairs(sel, excluded_neighbors=0)
        self.feat.add_distances_ca(periodic=False, excluded_neighbors=0)  # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs.shape[0])
        X = self.traj.xyz[:, pairs[:, 0], :]
        Y = self.traj.xyz[:, pairs[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(D, self.feat.transform(self.traj)))

    def test_ca_distances_with_all_atom_geometries(self):
        feat = MDFeaturizer(pdbfile_ops_aa)
        feat.add_distances_ca(excluded_neighbors=0)
        D_aa = feat.transform(mdtraj.load(pdbfile_ops_aa))

        # Create a reference
        feat_just_ca = MDFeaturizer(pdbfile_ops_Ca)
        feat_just_ca.add_distances(np.arange(feat_just_ca.topology.n_atoms))
        D_ca = feat_just_ca.transform(mdtraj.load(pdbfile_ops_Ca))
        assert(np.allclose(D_aa, D_ca))

    def test_ca_distances_with_all_atom_geometries_and_exclusions(self):
        feat = MDFeaturizer(pdbfile_ops_aa)
        feat.add_distances_ca(excluded_neighbors=2)
        D_aa = feat.transform(mdtraj.load(pdbfile_ops_aa))

        # Create a reference
        feat_just_ca = MDFeaturizer(pdbfile_ops_Ca)
        ca_pairs = feat.pairs(feat_just_ca.select_Ca(),excluded_neighbors=2)
        feat_just_ca.add_distances(ca_pairs)
        D_ca = feat_just_ca.transform(mdtraj.load(pdbfile_ops_Ca))
        assert(np.allclose(D_aa, D_ca))

    def test_ca_distances_with_residues_not_containing_cas_no_exclusions(self):
        # Load test geom
        geom = mdtraj.load(self.pdbfile)
        # No exclusions
        feat_EN0 = MDFeaturizer(self.bogus_geom_pdbfile)
        feat_EN0.add_distances_ca(excluded_neighbors=0)
        ENO_pairs = [[1,3],[1,5],[1,7],
                     [3,5], [3,7],
                     [5,7]
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
        EN1_pairs = [[1,3],[1,5],[1,7],
                     [3,5], [3,7],
                     [5,7]
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
        EN2_pairs = [[1,5],[1,7],
                     [3,7],
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
        EN1_pairs = [[1,3],[1,5],[1,7],
                     [3,5], [3,7],
                     [5,7]
                     ]
        assert (np.allclose(EN1_pairs, feat_EN1.active_features[0].distance_indexes))
        D = mdtraj.compute_distances(geom, EN1_pairs)
        assert (np.allclose(D, feat_EN1.transform(geom)))

    def test_contacts(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel, excluded_neighbors=2)
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
        assert(np.allclose(C, self.feat.transform(self.traj)))

    def test_contacts_count_contacts(self):
        sel = np.array([1, 2, 5, 20], dtype=int)
        pairs_expected = np.array([[1, 5], [1, 20], [2, 5], [2, 20], [5, 20]])
        pairs = self.feat.pairs(sel, excluded_neighbors=2)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_contacts(pairs, threshold=0.5, periodic=False, count_contacts=True)  # unperiodic distances such that we can compare
        # The dimensionality of the feature is now one
        assert(self.feat.dimension() == 1)
        X = self.traj.xyz[:, pairs_expected[:, 0], :]
        Y = self.traj.xyz[:, pairs_expected[:, 1], :]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        C = np.zeros(D.shape)
        I = np.argwhere(D <= 0.5)
        C[I[:, 0], I[:, 1]] = 1.0
        # Count the contacts
        C = C.sum(1, keepdims=True)
        assert(np.allclose(C, self.feat.transform(self.traj)))

    def test_angles(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())


    def test_angles_deg(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel, deg=True)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))

    def test_angles_cossin(self):
        sel = np.array([[1, 2, 5],
                        [1, 3, 8],
                        [2, 9, 10]], dtype=int)
        self.feat.add_angles(sel, cossin=True)
        assert(self.feat.dimension() == 2 * sel.shape[0])
        Y = self.feat.transform(self.traj)
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
        Y = self.feat.transform(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())

    def test_dihedrals_deg(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel, deg=True)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))
        self.assertEqual(len(self.feat.describe()), self.feat.dimension())

    def test_dihedrials_cossin(self):
        sel = np.array([[1, 2, 5, 6],
                        [1, 3, 8, 9],
                        [2, 9, 10, 12]], dtype=int)
        self.feat.add_dihedrals(sel, cossin=True)
        assert(self.feat.dimension() == 2 * sel.shape[0])
        Y = self.feat.transform(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrals(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_backbone_torsions()

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

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
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrals_cossin(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_backbone_torsions(cossin=True)

        traj = mdtraj.load(self.asn_leu_traj, top=self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
        self.assertEqual(Y.shape, (len(traj), 3*4)) # (3 phi + 3 psi)*2 [cos, sin]
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension(), msg=desc)
        self.assertIn("COS", desc[0])
        self.assertIn("SIN", desc[1])

    def test_backbone_dihedrials_chi(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_chi1_torsions()

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))
        desc = self.feat.describe()
        self.assertEqual(len(desc), self.feat.dimension())

    def test_backbone_dihedrials_chi_cossin(self):
        self.feat = MDFeaturizer(topfile=self.asn_leu_pdbfile)
        self.feat.add_chi1_torsions(cossin=True)

        traj = mdtraj.load(self.asn_leu_pdbfile)
        Y = self.feat.transform(traj)
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
        test_Y  = self.feat.transform(self.traj).squeeze()
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame])
        verbose_assertion_minrmsd(ref_Y, test_Y, self)
        assert self.feat.dimension() == 2
        assert len(self.feat.describe())==2

    def test_MinRmsd_with_atom_indices(self):
        # Test the Trajectory-input variant
        self.feat.add_minrmsd_to_ref(self.traj[self.ref_frame], atom_indices=self.atom_indices)
        # and the file-input variant
        self.feat.add_minrmsd_to_ref(xtcfile, ref_frame=self.ref_frame, atom_indices=self.atom_indices)
        test_Y  = self.feat.transform(self.traj).squeeze()
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame], atom_indices=self.atom_indices)
        verbose_assertion_minrmsd(ref_Y, test_Y, self)
        assert self.feat.dimension() == 2
        assert len(self.feat.describe())==2

    def test_MinRmsd_with_atom_indices_precentered(self):
        # Test the Trajectory-input variant
        self.feat.add_minrmsd_to_ref(self.traj[self.ref_frame], atom_indices=self.atom_indices, precentered=True)
        # and the file-input variant
        self.feat.add_minrmsd_to_ref(xtcfile, ref_frame=self.ref_frame, atom_indices=self.atom_indices, precentered=True)
        test_Y  = self.feat.transform(self.traj).squeeze()
        # now the reference
        ref_Y = mdtraj.rmsd(self.traj, self.traj[self.ref_frame], atom_indices=self.atom_indices, precentered=True)
        verbose_assertion_minrmsd(ref_Y, test_Y, self)
        assert self.feat.dimension() == 2
        assert len(self.feat.describe())==2

    def test_Residue_Mindist_Ca_all(self):
        n_ca = self.feat.topology.n_atoms
        self.feat.add_residue_mindist(scheme='ca')
        D = self.feat.transform(self.traj)
        Dref = mdtraj.compute_contacts(self.traj, scheme='ca')[0]
        assert np.allclose(D, Dref)
        assert len(self.feat.describe())==self.feat.dimension()

    def test_Residue_Mindist_Ca_all_threshold(self):
        threshold = .7
        self.feat.add_residue_mindist(scheme='ca', threshold=threshold)
        D = self.feat.transform(self.traj)
        Dref = mdtraj.compute_contacts(self.traj, scheme='ca')[0]
        Dbinary = np.zeros_like(Dref)
        I = np.argwhere(Dref <= threshold)
        Dbinary[I[:, 0], I[:, 1]] = 1
        assert np.allclose(D, Dbinary)
        assert len(self.feat.describe())==self.feat.dimension()

    def test_Residue_Mindist_Ca_array(self):
        contacts=np.array([[20,10,], [10,0]])
        self.feat.add_residue_mindist(scheme='ca', residue_pairs=contacts)
        D = self.feat.transform(self.traj)
        Dref = mdtraj.compute_contacts(self.traj, scheme='ca', contacts=contacts)[0]
        assert np.allclose(D, Dref)
        assert len(self.feat.describe())==self.feat.dimension()

    def test_Residue_Mindist_Ca_array_periodic(self):
        traj = mdtraj.load(pdbfile)
        # Atoms most far appart in Z
        atom_minz = traj.xyz.argmin(1).squeeze()[-1]
        atom_maxz = traj.xyz.argmax(1).squeeze()[-1]
        # Residues with the atoms most far appart in Z
        res_minz = traj.topology.atom(atom_minz).residue.index
        res_maxz = traj.topology.atom(atom_maxz).residue.index
        contacts=np.array([[res_minz, res_maxz]])
        # Tweak the trajectory so that a (bogus) PBC exists (otherwise traj._have_unitcell is False)
        traj.unitcell_angles = [90,90,90]
        traj.unitcell_lengths = [1, 1, 1]
        self.feat.add_residue_mindist(scheme='ca', residue_pairs=contacts, periodic=False)
        D = self.feat.transform(traj)
        Dperiodic_true  = mdtraj.compute_contacts(traj, scheme='ca', contacts=contacts, periodic=True)[0]
        Dperiodic_false = mdtraj.compute_contacts(traj, scheme='ca', contacts=contacts, periodic=False)[0]
        # This asserts that the periodic option is having an effect at all
        assert not np.allclose(Dperiodic_false, Dperiodic_true, )
        # This asserts that the periodic option is being handled correctly by pyemma
        assert np.allclose(D, Dperiodic_false)
        assert len(self.feat.describe())==self.feat.dimension()

    def test_Group_Mindist_One_Group(self):
        group0= [0,20,30,0]
        self.feat.add_group_mindist(group_definitions=[group0]) # Even with duplicates
        D = self.feat.transform(self.traj)
        dist_list = list(combinations(np.unique(group0),2))
        Dref = mdtraj.compute_distances(self.traj, dist_list)
        assert np.allclose(D.squeeze(), Dref.min(1))
        assert len(self.feat.describe())==self.feat.dimension()

    def test_Group_Mindist_All_Three_Groups(self):
        group0 = [0,20,30,0]
        group1 = [1,21,31,1]
        group2 = [2,22,32,2]
        self.feat.add_group_mindist(group_definitions=[group0, group1, group2])
        D = self.feat.transform(self.traj)

        # Now the references, computed separately for each combination of groups
        dist_list_01 = np.array(list(product(np.unique(group0),np.unique(group1))))
        dist_list_02 = np.array(list(product(np.unique(group0),np.unique(group2))))
        dist_list_12 = np.array(list(product(np.unique(group1),np.unique(group2))))
        Dref_01 = mdtraj.compute_distances(self.traj, dist_list_01).min(1)
        Dref_02 = mdtraj.compute_distances(self.traj, dist_list_02).min(1)
        Dref_12 = mdtraj.compute_distances(self.traj, dist_list_12).min(1)
        Dref = np.vstack((Dref_01,Dref_02,Dref_12)).T

        assert np.allclose(D.squeeze(), Dref)
        assert len(self.feat.describe())==self.feat.dimension()

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
        assert len(self.feat.describe())==self.feat.dimension()

    def test_Group_Mindist_Some_Three_Groups(self):
        group0 = [0,20,30,0]
        group1 = [1,21,31,1]
        group2 = [2,22,32,2]

        group_pairs=np.array([[0,1],
                              [2,2],
                              [0,2]])

        self.feat.add_group_mindist(group_definitions=[group0, group1, group2], group_pairs=group_pairs)
        D = self.feat.transform(self.traj)

        # Now the references, computed separately for each combination of groups
        dist_list_01 = np.array(list(product(np.unique(group0),np.unique(group1))))
        dist_list_02 = np.array(list(product(np.unique(group0),np.unique(group2))))
        dist_list_22 = np.array(list(combinations(np.unique(group2),2)))
        Dref_01 = mdtraj.compute_distances(self.traj, dist_list_01).min(1)
        Dref_02 = mdtraj.compute_distances(self.traj, dist_list_02).min(1)
        Dref_22 = mdtraj.compute_distances(self.traj, dist_list_22).min(1)
        Dref = np.vstack((Dref_01,Dref_22,Dref_02)).T

        assert np.allclose(D.squeeze(), Dref)
        assert len(self.feat.describe())==self.feat.dimension()


class TestFeaturizerNoDubs(unittest.TestCase):

    def testAddFeaturesWithDuplicates(self):
        """this tests adds multiple features twice (eg. same indices) and
        checks whether they are rejected or not"""
        featurizer = MDFeaturizer(pdbfile)
        expected_active = 1

        featurizer.add_angles([[0, 1, 2], [0, 3, 4]])
        featurizer.add_angles([[0, 1, 2], [0, 3, 4]])
        self.assertEqual(len(featurizer.active_features), expected_active)

        featurizer.add_contacts([[0, 1], [0, 3]])
        expected_active += 1
        self.assertEqual(len(featurizer.active_features), expected_active)
        featurizer.add_contacts([[0, 1], [0, 3]])
        self.assertEqual(len(featurizer.active_features), expected_active)

        # try to fool it with ca selection
        ca = featurizer.select_Ca()
        ca = featurizer.pairs(ca, excluded_neighbors=0)
        featurizer.add_distances(ca)
        expected_active += 1
        self.assertEqual(len(featurizer.active_features), expected_active)
        featurizer.add_distances_ca(excluded_neighbors=0)
        self.assertEqual(len(featurizer.active_features), expected_active)

        featurizer.add_inverse_distances([[0, 1], [0, 3]])
        expected_active += 1
        self.assertEqual(len(featurizer.active_features), expected_active)

        featurizer.add_distances([[0, 1], [0, 3]])
        expected_active += 1
        self.assertEqual(len(featurizer.active_features), expected_active)
        featurizer.add_distances([[0, 1], [0, 3]])
        self.assertEqual(len(featurizer.active_features), expected_active)

        def my_func(x):
            return x - 1

        def foo(x):
            return x - 1

        expected_active += 1
        my_feature = CustomFeature(my_func)
        my_feature.dimension = 3
        featurizer.add_custom_feature(my_feature)

        self.assertEqual(len(featurizer.active_features), expected_active)
        featurizer.add_custom_feature(my_feature)
        self.assertEqual(len(featurizer.active_features), expected_active)

        # since myfunc and foo are different functions, it should be added
        expected_active += 1
        foo_feat = CustomFeature(foo, dim=3)
        featurizer.add_custom_feature(foo_feat)

        self.assertEqual(len(featurizer.active_features), expected_active)

        expected_active += 1
        ref = mdtraj.load(xtcfile, top=pdbfile)
        featurizer.add_minrmsd_to_ref(ref)
        featurizer.add_minrmsd_to_ref(ref)
        self.assertEqual(len(featurizer.active_features), expected_active)

        expected_active += 1
        featurizer.add_minrmsd_to_ref(pdbfile)
        featurizer.add_minrmsd_to_ref(pdbfile)
        self.assertEqual(len(featurizer.active_features), expected_active)

        expected_active += 1
        featurizer.add_residue_mindist()
        featurizer.add_residue_mindist()
        self.assertEqual(len(featurizer.active_features), expected_active)

        expected_active += 1
        featurizer.add_group_mindist([[0,1],[0,2]])
        featurizer.add_group_mindist([[0,1],[0,2]])
        self.assertEqual(len(featurizer.active_features), expected_active)

    def test_labels(self):
        """ just checks for exceptions """
        featurizer = MDFeaturizer(pdbfile)
        featurizer.add_angles([[1, 2, 3], [4, 5, 6]])
        with self.assertRaises(ValueError) as cm:
            featurizer.add_backbone_torsions()
            assert 'emtpy indices' in cm.exception.message
        featurizer.add_contacts([[0, 1], [0, 3]])
        featurizer.add_distances([[0, 1], [0, 3]])
        featurizer.add_inverse_distances([[0, 1], [0, 3]])
        cs = CustomFeature(lambda x: x - 1, dim=3)
        featurizer.add_custom_feature(cs)
        featurizer.add_minrmsd_to_ref(pdbfile)
        featurizer.add_residue_mindist()
        featurizer.add_group_mindist([[0,1],[0,2]])

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
        str2 = _describe_atom(self.traj.topology,self.traj.n_atoms-1)
        assert len(str1.split()) >=4
        assert len(str2.split()) >=4
        assert str1.split()[-1] == '0'
        assert str2.split()[-1] == '1'

class TestStaticMethods(unittest.TestCase):

    def setUp(self):
        self.feat = MDFeaturizer(pdbfile)

    def test_pairs(self):
        n_at = 5
        pairs = self.feat.pairs(np.arange(n_at), excluded_neighbors=3)
        assert np.allclose(pairs, [0,4])

        pairs = self.feat.pairs(np.arange(n_at), excluded_neighbors=2)
        assert np.allclose(pairs, [[0,3],[0,4],
                                   [1,4]])

        pairs = self.feat.pairs(np.arange(n_at), excluded_neighbors=1)
        assert np.allclose(pairs, [[0,2], [0,3],[0,4],
                                   [1,3], [1,4],
                                   [2,4]])

        pairs = self.feat.pairs(np.arange(n_at), excluded_neighbors=0)
        assert np.allclose(pairs, [[0,1], [0,2], [0,3],[0,4],
                                   [1,2], [1,3], [1,4],
                                   [2,3], [2,4],
                                   [3,4]])

# Define some function that somehow mimics one would typically want to do,
# e.g. 1. call mdtraj,
#      2. perform some other operations on the result
#      3. return a numpy array
def some_call_to_mdtraj_some_operations_some_linalg(traj, pairs, means, U):
    D = mdtraj.compute_distances(traj, pairs)
    D_meanfree =  D - means
    Y = (U.T.dot(D_meanfree.T)).T
    return Y.astype('float32')

class TestCustomFeature(unittest.TestCase):

    def setUp(self):
        self.feat = MDFeaturizer(pdbfile)
        self.traj = mdtraj.load(xtcfile, top=pdbfile)


        self.pairs = [[0,1],[0,2], [1,2]]           #some distances
        self.means = [.5, .75, 1.0]               #bogus means
        self.U = np.array([[0,1],
                           [1,0],
                           [1,1]])           #bogus transformation, projects from 3 distances to 2 components
    def test_some_feature(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg   , self.U.shape[1],
                                        self.pairs,
                                        self.means,
                                        self.U
                                        )

        Y_custom_feature = self.feat.transform(self.traj)
        # Directly call the function
        Y_function =  some_call_to_mdtraj_some_operations_some_linalg(self.traj, self.pairs, self.means, self.U)
        assert np.allclose(Y_custom_feature, Y_function)

    def test_describe(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1],
                                  self.pairs,
                                  self.means,
                                  self.U
                                  )
        self.feat.describe()

    def test_dimensionality(self):
        self.feat.add_custom_func(some_call_to_mdtraj_some_operations_some_linalg, self.U.shape[1],
                                  self.pairs,
                                  self.means,
                                  self.U
                                  )

        assert self.feat.dimension()==self.U.shape[1]

if __name__ == "__main__":
    unittest.main()
