import unittest
import numpy as np

import os
import mdtraj

from pyemma.coordinates.io import featurizer as ft
from pyemma.coordinates.io.featurizer import MDFeaturizer, CustomFeature
from pyemma.coordinates.tests.test_discretizer import create_water_topology_on_disc

path = os.path.join(os.path.split(__file__)[0],'data')
xtcfile = os.path.join(path, 'bpti_mini.xtc')
pdbfile = os.path.join(path, 'bpti_ca.pdb')


class TestFeaturizer(unittest.TestCase):

    def setUp(self):
        self.pdbfile = pdbfile
        self.traj = mdtraj.load(xtcfile, top=self.pdbfile)
        self.feat = MDFeaturizer(self.pdbfile)

    def test_select_all(self):
        self.feat.add_all()
        assert (self.feat.dimension() == self.traj.n_atoms * 3)
        refmap = np.reshape(self.traj.xyz, (len(self.traj), self.traj.n_atoms * 3))
        assert (np.all(refmap == self.feat.map(self.traj)))

    def test_select(self):
        sel = np.array([1,2,5,20], dtype=int)
        self.feat.add_selection(sel)
        assert (self.feat.dimension() == sel.shape[0] * 3)
        refmap = np.reshape(self.traj.xyz[:,sel,:], (len(self.traj), sel.shape[0] * 3))
        assert (np.all(refmap == self.feat.map(self.traj)))

    def test_distances(self):
        sel = np.array([1,2,5,20], dtype=int)
        pairs_expected = np.array([[1,5],[1,20],[2,5],[2,20],[5,20]])
        pairs = self.feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_distances(pairs, periodic=False) # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:,pairs_expected[:,0],:]
        Y = self.traj.xyz[:,pairs_expected[:,1],:]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(D, self.feat.map(self.traj)))

    def test_inverse_distances(self):
        sel = np.array([1,2,5,20], dtype=int)
        pairs_expected = np.array([[1,5],[1,20],[2,5],[2,20],[5,20]])
        pairs = self.feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_inverse_distances(pairs, periodic=False) # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:,pairs_expected[:,0],:]
        Y = self.traj.xyz[:,pairs_expected[:,1],:]
        Dinv = 1.0/np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(Dinv, self.feat.map(self.traj)))

    def test_ca_distances(self):
        sel = self.feat.select_Ca()
        assert(np.all(sel == range(self.traj.n_atoms))) # should be all for this Ca-traj
        pairs = self.feat.pairs(sel)
        self.feat.add_distances_ca(periodic=False) # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs.shape[0])
        X = self.traj.xyz[:,pairs[:,0],:]
        Y = self.traj.xyz[:,pairs[:,1],:]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        assert(np.allclose(D, self.feat.map(self.traj)))

    def test_contacts(self):
        sel = np.array([1,2,5,20], dtype=int)
        pairs_expected = np.array([[1,5],[1,20],[2,5],[2,20],[5,20]])
        pairs = self.feat.pairs(sel)
        assert(pairs.shape == pairs_expected.shape)
        assert(np.all(pairs == pairs_expected))
        self.feat.add_contacts(pairs, threshold=0.5, periodic=False) # unperiodic distances such that we can compare
        assert(self.feat.dimension() == pairs_expected.shape[0])
        X = self.traj.xyz[:,pairs_expected[:,0],:]
        Y = self.traj.xyz[:,pairs_expected[:,1],:]
        D = np.sqrt(np.sum((X - Y) ** 2, axis=2))
        C = np.zeros(D.shape)
        I = np.argwhere(D <= 0.5)
        C[I[:,0],I[:,1]] = 1.0
        assert(np.allclose(C, self.feat.map(self.traj)))

    def test_angles(self):
        sel = np.array([[1,2,5],
                        [1,3,8],
                        [2,9,10]], dtype=int)
        self.feat.add_angles(sel)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

    def test_angles_deg(self):
        sel = np.array([[1,2,5],
                        [1,3,8],
                        [2,9,10]], dtype=int)
        self.feat.add_angles(sel, deg=True)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))

    def test_dihedrals(self):
        sel = np.array([[1,2,5,6],
                        [1,3,8,9],
                        [2,9,10,12]], dtype=int)
        self.feat.add_dihedrals(sel)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -np.pi))
        assert(np.alltrue(Y <= np.pi))

    def test_dihedrals_deg(self):
        sel = np.array([[1,2,5,6],
                        [1,3,8,9],
                        [2,9,10,12]], dtype=int)
        self.feat.add_dihedrals(sel, deg=True)
        assert(self.feat.dimension() == sel.shape[0])
        Y = self.feat.map(self.traj)
        assert(np.alltrue(Y >= -180.0))
        assert(np.alltrue(Y <= 180.0))

    def test_backbone_dihedrals(self):
        # TODO: test me
        pass

    def test_backbone_dihedrals_deg(self):
        # TODO: test me
        pass

    def test_custom_feature(self):
        # TODO: test me
        pass


class TestFeaturizerNoDubs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFeaturizerNoDubs, cls).setUpClass()
        cls.old_lvl = ft.log.level
        ft.log.level = 50

    @classmethod
    def tearDownClass(cls):
        ft.log.level = cls.old_lvl

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

        featurizer.describe()


if __name__ == "__main__":
    unittest.main()
