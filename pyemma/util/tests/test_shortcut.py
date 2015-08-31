
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.



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