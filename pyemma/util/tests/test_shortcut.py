from pyemma.util.annotators import shortcut
import unittest

@shortcut("bar")
def foo():
    """ sample docstring"""
    return True

class TestShortcut(unittest.TestCase):

    def test_shortcut(self):
        self.assertEqual( bar.__doc__ , foo.__doc__)
        result = bar()
        self.assertTrue(result)
