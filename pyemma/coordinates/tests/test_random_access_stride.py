import os
import unittest
from unittest import TestCase
import numpy as np
import pyemma.coordinates.api as coor
import pkg_resources


class TestRandomAccessStride(TestCase):
    def setUp(self):
        self.dim = 5
        self.data = [np.random.random((100, self.dim)),
                     np.random.random((1, self.dim)),
                     np.random.random((2, self.dim))]
        self.stride = {0: [1, 2, 3], 2: [1]}

    def test_data_in_memory_random_access(self):
        # TODO: lagged?
        # access with a chunk_size that is larger than the largest index list of stride
        data_in_memory = coor.source(self.data, chunk_size=10)
        out1 = data_in_memory.get_output(stride=self.stride)

        # access with a chunk_size that is smaller than the largest index list of stride
        data_in_memory = coor.source(self.data, chunk_size=1)
        out2 = data_in_memory.get_output(stride=self.stride)

        # access in full trajectory mode
        data_in_memory = coor.source(self.data, chunk_size=0)
        out3 = data_in_memory.get_output(stride=self.stride)

        for idx in self.stride.keys():
            np.testing.assert_array_equal(out1[idx], out2[idx])
            np.testing.assert_array_equal(out2[idx], out3[idx])

    def test_transformer_iterator_random_access(self):
        # TODO: lagged?
        kmeans = coor.cluster_kmeans(self.data, k=2)
        kmeans.in_memory = True

        for cs in xrange(1, 5):
            kmeans.chunksize = cs
            ref_stride = {0: 0, 1: 0, 2: 0}
            for x in kmeans.iterator(stride=self.stride):
                ref_stride[x[0]] += len(x[1])
            for key in ref_stride.keys():
                expected = (len(self.stride[key]) if key in self.stride.keys() else 0)
                assert ref_stride[key] == expected, \
                    "Expected to get exactly %s elements of trajectory %s, but got %s for chunksize=%s" \
                    % (expected, key, ref_stride[key], cs)

    def test_feature_reader_random_access(self):
        from pyemma.coordinates.tests.test_featurereader import create_traj
        topfile = pkg_resources.resource_filename('pyemma.coordinates.tests.test_featurereader', 'data/test.pdb')
        trajfile = None
        try:
            trajfile, xyz, nframes = create_traj(topfile)
            source = coor.source(trajfile, top=topfile)

            out = source.get_output(stride=self.stride)
            print out
        finally:
            if trajfile:
                try:
                    os.unlink(trajfile)
                except EnvironmentError:
                    pass


if __name__ == '__main__':
    unittest.main()
