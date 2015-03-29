import unittest
import numpy as np

import os
import mdtraj

from pyemma.coordinates.io.featurizer import MDFeaturizer

class TestFeaturizer(unittest.TestCase):

    def setUp(self):
        path = os.path.join(os.path.split(__file__)[0],'data')
        xtcfile = os.path.join(path, 'bpti_mini.xtc')
        self.pdbfile = os.path.join(path, 'bpti_ca.pdb')
        self.traj = mdtraj.load(xtcfile, top=self.pdbfile)

    def test_select_all(self):
        feat = MDFeaturizer(self.pdbfile)
        feat.add_all()
        assert (feat.dimension == self.traj.n_atoms * 3)
        refmap = np.reshape(self.traj.xyz, (len(self.traj), self.traj.n_atoms * 3))
        assert (np.all(refmap == feat.map(self.traj)))

    def test_select(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = np.array([1,2,5,20], dtype=int)
        feat.add_selection(sel)
        assert (feat.dimension == sel.shape[0] * 3)
        refmap = np.reshape(self.traj.xyz[:,sel,:], (len(self.traj), sel.shape[0] * 3))
        assert (np.all(refmap == feat.map(self.traj)))

    def test_distances(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = np.array([1,2,5,20], dtype=int)
        pairs_expected = np.array([[1,5],[1,20],[2,5],[2,20],[5,20]])
        pairs = feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        feat.add_distances(pairs, periodic=False) # unperiodic distances such that we can compare
        assert(feat.dimension == pairs_expected.shape[0])
        X = self.traj.xyz[:,pairs_expected[:,0],:]
        Y = self.traj.xyz[:,pairs_expected[:,1],:]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(D, feat.map(self.traj)))

    def test_inverse_distances(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = np.array([1,2,5,20], dtype=int)
        pairs_expected = np.array([[1,5],[1,20],[2,5],[2,20],[5,20]])
        pairs = feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        feat.add_inverse_distances(pairs, periodic=False) # unperiodic distances such that we can compare
        assert(feat.dimension == pairs_expected.shape[0])
        X = self.traj.xyz[:,pairs_expected[:,0],:]
        Y = self.traj.xyz[:,pairs_expected[:,1],:]
        Dinv = 1.0/np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(Dinv, feat.map(self.traj)))

    def test_ca_distances(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = feat.select_Ca()
        assert(np.all(sel == range(self.traj.n_atoms))) # should be all for this Ca-traj
        pairs = feat.pairs(sel)
        feat.add_distances_ca(periodic=False) # unperiodic distances such that we can compare
        assert(feat.dimension == pairs.shape[0])
        X = self.traj.xyz[:,pairs[:,0],:]
        Y = self.traj.xyz[:,pairs[:,1],:]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(D, feat.map(self.traj)))

    def test_contacts(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = np.array([1,2,5,20], dtype=int)
        pairs_expected = np.array([[1,5],[1,20],[2,5],[2,20],[5,20]])
        pairs = feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        feat.add_contacts(pairs, threshold=0.5, periodic=False) # unperiodic distances such that we can compare
        assert(feat.dimension == pairs_expected.shape[0])
        X = self.traj.xyz[:,pairs_expected[:,0],:]
        Y = self.traj.xyz[:,pairs_expected[:,1],:]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        C = np.zeros(D.shape)
        I = np.argwhere(D <= 0.5)
        C[I[:,0],I[:,1]] = 1.0
        assert(np.allclose(C, feat.map(self.traj)))

    def test_angles(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = np.array([[1,2,5],
                        [1,3,8],
                        [2,9,10]], dtype=int)
        feat.add_angles(sel)
        assert(feat.dimension == sel.shape[0])
        Y = feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

    def test_angles_deg(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = np.array([[1,2,5],
                        [1,3,8],
                        [2,9,10]], dtype=int)
        feat.add_angles(sel, deg=True)
        assert(feat.dimension == sel.shape[0])
        Y = feat.map(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))

    def test_dihedrals(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = np.array([[1,2,5,6],
                        [1,3,8,9],
                        [2,9,10,12]], dtype=int)
        feat.add_dihedrals(sel)
        assert(feat.dimension == sel.shape[0])
        Y = feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

    def test_dihedrals_deg(self):
        feat = MDFeaturizer(self.pdbfile)
        sel = np.array([[1,2,5,6],
                        [1,3,8,9],
                        [2,9,10,12]], dtype=int)
        feat.add_dihedrals(sel, deg=True)
        assert(feat.dimension == sel.shape[0])
        Y = feat.map(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))
        
    def test_backbone_dihedrals(self):
        #TODO: test me
        pass

    def test_backbone_dihedrals_deg(self):
        #TODO: test me
        pass

    def test_custom_feature(self):
        #TODO: test me
        pass


if __name__ == "__main__":
    unittest.main()
