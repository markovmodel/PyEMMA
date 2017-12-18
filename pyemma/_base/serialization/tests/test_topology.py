import tempfile
import unittest

from pyemma._base.serialization.h5file import H5Wrapper


class TestTopology(unittest.TestCase):

    def test(self):
        import pkg_resources
        import mdtraj

        traj = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/opsin_aa_1_frame.pdb.gz')
        top = mdtraj.load(traj).top
        f = tempfile.mktemp('.h5')
        with H5Wrapper(f) as fh:
            fh.add_object('top', top)
            restored = fh.model

        assert top == restored
