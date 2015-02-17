'''
Created on 19.01.2015

@author: marscher
'''
import itertools
import os
import tempfile
import unittest
import mdtraj
import numpy as np

from mdtraj.core.trajectory import Trajectory
from mdtraj.core.element import hydrogen, oxygen
from mdtraj.core.topology import Topology

from pyemma.coordinates.clustering.uniform_time import UniformTimeClustering
from ..discretizer import Discretizer
from ..io.feature_reader import FeatureReader
from ..transform.pca import PCA


def create_water_topology_on_disc(n):
    topfile = tempfile.mkstemp('.pdb')[1]
    top = Topology()
    chain = top.add_chain()

    for i in xrange(n):
        res = top.add_residue('r%i' % i, chain)
        h1 = top.add_atom('H', hydrogen, res)
        o = top.add_atom('O', oxygen, res)
        h2 = top.add_atom('H', hydrogen, res)
        top.add_bond(h1, o)
        top.add_bond(h2, o)

    xyz = np.zeros((n * 3, 3))
    Trajectory(xyz, top).save_pdb(topfile)
    return topfile


def create_traj_on_disc(topfile, n_frames, n_atoms):
    fn = tempfile.mktemp('.xtc')
    xyz = np.random.random((n_frames, n_atoms, 3))
    t = mdtraj.load(topfile)
    t.xyz = xyz
    t.time = np.arange(n_frames)
    t.save(fn)
    return fn


class TestDiscretizer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        c = super(TestDiscretizer, cls).setUpClass()
        # create a fake trajectory which has 2 atoms and coordinates are just a range
        # over all frames.
        cls.n_frames = 1000
        cls.n_residues = 30
        cls.topfile = create_water_topology_on_disc(cls.n_residues)

        # create some trajectories
        t1 = create_traj_on_disc(
            cls.topfile, cls.n_frames, cls.n_residues * 3)

        t2 = create_traj_on_disc(
            cls.topfile, cls.n_frames, cls.n_residues * 3)

        cls.trajfiles = [t1, t2]

        return c

    @classmethod
    def tearDownClass(cls):
        """delete temporary files"""
        os.unlink(cls.topfile)
        for f in cls.trajfiles:
            os.unlink(f)

    def test(self):
        reader = FeatureReader(self.trajfiles, self.topfile)
        # select all possible distances
        pairs = np.array(
            [x for x in itertools.combinations(range(self.n_residues), 2)])

        reader.featurizer.distances(pairs)

        tica = PCA(output_dimension=2)

        n_clusters = 2
        clustering = UniformTimeClustering(k=n_clusters)

        D = Discretizer(reader, transform=tica, cluster=clustering)
        D.run()

        self.assertEqual(len(D.dtrajs), len(self.trajfiles))

        for dtraj in clustering.dtrajs:
            unique = np.unique(dtraj)
            self.assertEqual(unique.shape[0], n_clusters)


if __name__ == "__main__":
    unittest.main()
