import unittest
import pkg_resources
import os
from glob import glob
import numpy as np

from pyemma.coordinates import source
from pyemma.coordinates.data.joiner import Joiner


class TestJoiner(unittest.TestCase):

    def setUp(self):
        self.readers = []
        data_dir = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data')
        trajs = glob(data_dir + "/bpti_0*.xtc")
        top = os.path.join(data_dir, 'bpti_ca.pdb')
        self.readers.append(source(trajs, top=top))
        ndim = self.readers[0].ndim
        lengths = self.readers[0].trajectory_lengths()
        arrays = [np.random.random( (length, ndim) ) for length in lengths]

        self.desired_combined_output = None

        self.readers.append(source(arrays))

    def test_combined_output(self):
        j = Joiner(self.readers)
        out = j.get_output()
        assert len(out) == 3
        assert j.ndim == self.readers[0].ndim * 2
        np.testing.assert_equal(j.trajectory_lengths(), self.readers[0].trajectory_lengths())

        from collections import defaultdict
        outs = defaultdict(list)
        for r in self.readers:
            for i, x in enumerate(r.get_output()):
                outs[i].append(x)
        combined = [np.hstack(outs[i]) for i in range(3)]
        np.testing.assert_equal(out, combined)

