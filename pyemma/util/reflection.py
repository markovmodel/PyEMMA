
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

from __future__ import division, print_function, absolute_import

import inspect
import six
#from six import string_types
from collections import namedtuple

__author__ = 'noe, marscher'


# Add a replacement for inspect.getargspec() which is deprecated in python 3.5
# The version below is borrowed from Django,
# https://github.com/django/django/pull/4846

# Note an inconsistency between inspect.getargspec(func) and
# inspect.signature(func). If `func` is a bound method, the latter does *not* 
# list `self` as a first argument, while the former *does*.
# Hence cook up a common ground replacement: `getargspec_no_self` which
# mimics `inspect.getargspec` but does not list `self`.
#
# This way, the caller code does not need to know whether it uses a legacy
# .getargspec or bright and shiny .signature.

try:
    # is it python 3.3 or higher?
    inspect.signature

    # Apparently, yes. Wrap inspect.signature

    ArgSpec = namedtuple('ArgSpec', ['args', 'varargs', 'keywords', 'defaults'])

    def getargspec_no_self(func):
        """inspect.getargspec replacement using inspect.signature.

        inspect.getargspec is deprecated in python 3. This is a replacement
        based on the (new in python 3.3) `inspect.signature`.

        Parameters
        ----------
        func : callable
            A callable to inspect

        Returns
        -------
        argspec : ArgSpec(args, varargs, varkw, defaults)
            This is similar to the result of inspect.getargspec(func) under
            python 2.x.
            NOTE: if the first argument of `func` is self, it is *not*, I repeat
            *not* included in argspec.args.
            This is done for consistency between inspect.getargspec() under
            python 2.x, and inspect.signature() under python 3.x.
        """
        sig = inspect.signature(func)
        args = [
            p.name for p in sig.parameters.values()
            if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        varargs = [
            p.name for p in sig.parameters.values()
            if p.kind == inspect.Parameter.VAR_POSITIONAL
        ]
        varargs = varargs[0] if varargs else None
        varkw = [
            p.name for p in sig.parameters.values()
            if p.kind == inspect.Parameter.VAR_KEYWORD
        ]
        varkw = varkw[0] if varkw else None
        defaults = [
            p.default for p in sig.parameters.values()
            if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
               p.default is not p.empty)
        ] or None

        if args[0] == 'self':
            args.pop(0)

        return ArgSpec(args, varargs, varkw, defaults)

except AttributeError:
    # python 2.x
    def getargspec_no_self(func):
        """inspect.getargspec replacement for compatibility with python 3.x.

        inspect.getargspec is deprecated in python 3. This wraps it, and
        *removes* `self` from the argument list of `func`, if present.
        This is done for forward compatibility with python 3.

        Parameters
        ----------
        func : callable
            A callable to inspect

        Returns
        -------
        argspec : ArgSpec(args, varargs, varkw, defaults)
            This is similar to the result of inspect.getargspec(func) under
            python 2.x.
            NOTE: if the first argument of `func` is self, it is *not*, I repeat
            *not* included in argspec.args.
            This is done for consistency between inspect.getargspec() under
            python 2.x, and inspect.signature() under python 3.x.
        """
        argspec = inspect.getargspec(func)
        if argspec.args[0] == 'self':
            argspec.args.pop(0)
        return argspec


def call_member(obj, f, *args, **kwargs):
    """ Calls the specified method, property or attribute of the given object

    Parameters
    ----------
    obj : object
        The object that will be used
    f : str or function
        Name of or reference to method, property or attribute
    failfast : bool
        If True, will raise an exception when trying a method that doesn't exist. If False, will simply return None
        in that case
    """
    # get function name
    if not isinstance(f, six.string_types):
        fname = f.__func__.__name__
    else:
        fname = f
    # get the method ref
    method = getattr(obj, fname)
    # handle cases
    if inspect.ismethod(method):
        return method(*args, **kwargs)

    # attribute or property
    return method


def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    args, varargs, keywords, defaults = getargspec_no_self(func)
    return dict(zip(args[-len(defaults):], defaults))
