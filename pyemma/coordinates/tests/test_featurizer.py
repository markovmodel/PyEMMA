'''
Created on 27.03.2015

@author: marscher
'''
import unittest
import os

from pyemma.coordinates.io.featurizer import MDFeaturizer, CustomFeature
from pyemma.coordinates.tests.test_discretizer import create_water_topology_on_disc
from pyemma.coordinates.io import featurizer as ft


class TestFeaturizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super(TestFeaturizer, cls).setUpClass()
        cls.topfile = create_water_topology_on_disc(100)
        cls.old_lvl = ft.log.level
        ft.log.level = 50

    @classmethod
    def tearDownClass(cls):
        os.unlink(cls.topfile)
        ft.log.level = cls.old_lvl

    def testAddFeaturesWithDuplicates(self):
        """this tests adds multiple features twice (eg. same indices) and
        checks whether they are rejected or not"""
        featurizer = MDFeaturizer(self.topfile)

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
        featurizer.add_custom_feature(my_feature, 3)

        self.assertEqual(len(featurizer.active_features), 7)
        featurizer.add_custom_feature(my_feature, 3)
        self.assertEqual(len(featurizer.active_features), 7)
        # since myfunc and foo are different functions, it should be added
        featurizer.add_custom_feature(CustomFeature(foo), 3)
        self.assertEqual(len(featurizer.active_features), 8)

    def test_labels(self):
        """ just checks for exceptions """
        featurizer = MDFeaturizer(self.topfile)
        featurizer.add_angles([[1, 2, 3], [4, 5, 6]])
        featurizer.add_backbone_torsions()
        featurizer.add_contacts([[0, 1], [0, 3]])
        featurizer.add_distances([[0, 1], [0, 3]])
        featurizer.add_inverse_distances([[0, 1], [0, 3]])
        cs = CustomFeature(lambda x: x - 1)
        featurizer.add_custom_feature(cs, 3)

        featurizer.describe()

if __name__ == "__main__":
    unittest.main()
