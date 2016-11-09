import unittest

from pyemma.util.contexts import attribute

class TestContexts(unittest.TestCase):

    def test_attribute_context(self):
        class Foo:
            def __init__(self):
                self._x = 5

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = value
        foo = Foo()
        with attribute(foo, 'x', 10):
            assert foo.x == 10
        assert foo.x == 5

        try:
            with attribute(foo, 'x', 100):
                assert foo.x == 100
                raise RuntimeError()
        except RuntimeError:
            assert foo.x == 5
        assert foo.x == 5

if __name__ == "__main__":
    unittest.main()
