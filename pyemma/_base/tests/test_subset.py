import unittest

from pyemma._base.subset import add_full_state_methods, map_to_full_state, SubSet
import numpy as np

import types

k_global = 3


@add_full_state_methods
class test_class(SubSet):
    def __init__(self, active_set=(2, 3, 4), nstates_full=10):
        self.active_set = active_set
        self.nstates_full = nstates_full

    @map_to_full_state(default_arg=0)
    def eigenvalues(self, k=None):
        assert k == k_global, k
        return [0, 1, 2]

    @map_to_full_state(default_arg=np.inf)
    def right_eigenvectors(self):
        return np.arange(27).reshape(-1, 3)

    @map_to_full_state(default_arg=np.inf, extend_along_axis=1)
    def left_eigenvectors(self):
        return np.arange(27).reshape(-1, 3).T

    @property
    @map_to_full_state(default_arg=None)
    def test_property(self):
        return [4, 5, 6]


class TestSubset(unittest.TestCase):
    def test_has_member(self):
        inst = test_class()
        assert hasattr(inst, 'eigenvalues_full_state')
        assert hasattr(test_class, 'eigenvalues_full_state')
        self.assertIsInstance(inst.eigenvalues_full_state, types.MethodType)
        self.assertIsInstance(test_class.test_property, property)

    def test_ev(self):
        inst = test_class()
        expected = np.zeros(inst.nstates_full)
        expected[inst.active_set] = inst.eigenvalues(k=k_global)
        np.testing.assert_equal(inst.eigenvalues_full_state(k=k_global), expected)

    def test_shape_left_ev(self):
        inst = test_class(np.arange(start=23, stop=32), nstates_full=60)
        shape = inst.left_eigenvectors().shape
        result = inst.left_eigenvectors_full_state()
        expected = np.array([np.inf] * shape[0] * inst.nstates_full).reshape(shape[0], -1)
        expected[:, inst.active_set] = inst.left_eigenvectors()
        np.testing.assert_equal(expected, result)

    def test_shape_right_ev(self):
        inst = test_class(np.arange(start=3, stop=12), nstates_full=60)
        shape = inst.right_eigenvectors().shape
        result = inst.right_eigenvectors_full_state()
        expected = np.array([np.inf] * inst.nstates_full * shape[1]).reshape(-1, shape[1])
        expected[inst.active_set, :] = inst.right_eigenvectors()
        np.testing.assert_equal(expected, result)

if __name__ == '__main__':
    unittest.main()

