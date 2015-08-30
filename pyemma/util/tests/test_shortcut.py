

from __future__ import absolute_import
from pyemma.util.annotators import shortcut, aliased, alias
import unittest

@shortcut("bar", "bar2")
def foo():
    """ sample docstring"""
    return True

@aliased
class Foo(object):
    @alias("bar2", "bar3")
    def bar(self):
        """ doc """
        pass

class TestShortcut_and_Aliases(unittest.TestCase):

    def test_shortcut(self):
        self.assertEqual(bar.__doc__, foo.__doc__)
        result = bar()
        self.assertTrue(result)
        
        self.assertEqual(bar2.__doc__, foo.__doc__)
        result = bar2()
        self.assertTrue(result)

    def test_alias_class(self):
        assert hasattr(Foo, "bar3")
        assert hasattr(Foo, "bar3")

    def test_alias_class_inst(self):
        inst = Foo()
        assert hasattr(inst, "bar2")
        assert hasattr(inst, "bar3")
        self.assertEqual(inst.bar.__doc__, inst.bar2.__doc__)

