'''
Created on 23.07.2015

@author: marscher
'''
import unittest
import numpy as np
from pyemma.coordinates.data.data_in_memory import DataInMemory
from pyemma.coordinates.transform.sparsifier import Sparsifier
import pyemma.coordinates.api as coor_api
import mdtraj


class TestSparsifier(unittest.TestCase):

    def setUp(self):
        self.X = np.random.random((1000, 10))
        ones = np.ones((1000, 1))
        data = np.concatenate((self.X, ones), axis=1)
        self.src = DataInMemory(data)
        self.src.chunksize = 200

        self.sparsifier = Sparsifier()
        self.sparsifier.data_producer = self.src
        self.sparsifier.parametrize()

    def test_constant_column(self):
        out = self.sparsifier.get_output()[0]
        np.testing.assert_allclose(out, self.X)

    def test_constant_column_tica(self):
        tica_obj = coor_api.tica(
            self.sparsifier, kinetic_map=True, var_cutoff=1)
        self.assertEqual(tica_obj.dimension(), self.sparsifier.dimension())

    def test_numerical_close_features(self):
        rtol = self.sparsifier.rtol
        noise = (rtol * 200) * (np.random.random(1000) * 2 - 0.5)
        self.src._data[0][:, -1] += noise

        out = self.sparsifier.get_output()[0]
        np.testing.assert_allclose(out, self.X)


class TestSparsifierImplicitTICA(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """ 3 particles live on the x-axis, two of them have a constant contact,
        only one is moving; so the contact between 2nd and 3rd can be eliminated
        by prior diagonalisation in TICA."""

        cls.n = 3
        cls.n_frames = 200
        xyz = 3 * np.random.random((cls.n_frames, cls.n, 3))

        # force particles to only move on the x-axis
        xyz[:, :, 1] = 0
        xyz[:, :, 2] = 0

        # form a "fake permenant" contact between first and third particle
        first_particle_coords = xyz[0, 0, :]
        xyz[:, 2, :] = first_particle_coords + np.array([0.25, 0, 0])
        xyz[:, 1, :] = first_particle_coords

        from mdtraj import Trajectory, Topology
        from mdtraj.formats import PDBTrajectoryFile
        import tempfile

        cls.trajfile = tempfile.mktemp(suffix=".dcd")
        cls.topfile = tempfile.mktemp(suffix=".pdb")

        t = Topology()
        c = t.add_chain()
        for i in range(cls.n):
            res = t.add_residue("r%i" % i, c)
            t.add_atom('P%i' % i, element=None, residue=res)

        traj = Trajectory(xyz, t)

        traj.save_dcd(cls.trajfile)
        with PDBTrajectoryFile(cls.topfile, 'w') as f:
            f.write(xyz[0], t)

    @classmethod
    def tearDownClass(cls):
        import os
        try:
            os.unlink(cls.trajfile)
            os.unlink(cls.topfile)
        except EnvironmentError:
            pass

    def test_pipeline_tica_contact_features(self):
        # contact features are currently the only potentially constant features
        reader = coor_api.source(self.trajfile, top=self.topfile)
        self.assertEqual(reader.dimension(), self.n * 3)

        # form all contact pairs
        import itertools
        pairs = list(itertools.combinations(range(self.n), 2))
        reader.featurizer.add_contacts(pairs)

        sparsifier = Sparsifier()
        sparsifier.data_producer = reader
        sparsifier.parametrize()

        self.assertEqual(sparsifier.dimension(), reader.dimension() - 1)

        tica = coor_api.tica(reader, lag=1, var_cutoff=1)

        # internal trigger for sparse featurization should be turned on
        self.assertTrue(tica._has_potentially_sparse_output)
        # we have one constant column, so the sparsifier will remove this
        self.assertEqual(tica.dimension(), sparsifier.dimension())

        self.assertEqual(len(tica.cov), sparsifier.dimension())

    def test_pipeline_tica_contact_features_given_mean(self):
        reader = coor_api.source(self.trajfile, top=self.topfile)

        # form all contact pairs
        import itertools
        pairs = list(itertools.combinations(range(self.n), 2))
        reader.featurizer.add_contacts(pairs)

        # calculate mean ourselfs and pass it to tica, the api method should
        # map the given mean array to varying indices.
        mean = reader.get_output()[0].mean(axis=0).astype(np.float64)

        tica = coor_api.tica(reader, lag=1, var_cutoff=1, mean=mean)

        # internal trigger for sparse featurization should be turned on
        self.assertTrue(tica._has_potentially_sparse_output)
        # we have one constant column, so the sparsifier will remove this
        self.assertEqual(tica.dimension(), reader.dimension() - 1)


if __name__ == "__main__":
    unittest.main()
