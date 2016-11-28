import numpy as np
import scipy.linalg as scl
import unittest
import pkg_resources

from pyemma._ext.variational.solvers.direct import sort_by_norm
from pyemma.coordinates.estimation.koopman import _KoopmanEstimator
from pyemma.coordinates import source


class TestKoopman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Basis set definition:
        cls.nf = 10
        cls.chi = np.zeros((20, cls.nf), dtype=float)
        for n in range(cls.nf):
            cls.chi[2*n:2*(n+1), n] = 1.0

        # Load simulations:
        f = np.load(pkg_resources.resource_filename(__name__, "data/test_data_koopman.npz"))
        trajs = [f[key] for key in f.keys()]
        cls.data = [cls.chi[traj, :] for traj in trajs]
        cls.source_obj = source(cls.data)

        # Lag time:
        cls.tau = 10
        # Truncation for small eigenvalues:
        cls.epsilon = 1e-6
        # Chunksize:
        cls.chunksize = 200

        # Generate _KoopmanEstimator:
        Kest = _KoopmanEstimator(cls.tau, epsilon=cls.epsilon, chunksize=cls.chunksize)
        weight_object = Kest.weights


if __name__ == "__main__":
    unittest.main()