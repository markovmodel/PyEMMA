import tempfile
import unittest

from pyemma._base.serialization.serialization import save, load


class TestTopology(unittest.TestCase):

    def test(self):
        import pkg_resources
        import mdtraj

        traj = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/opsin_aa_1_frame.pdb.gz')
        top = mdtraj.load(traj).top
        f = tempfile.mktemp('.h5')
        save(top, f)
        restored = load(f)

        assert top == restored
