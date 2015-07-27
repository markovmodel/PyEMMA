# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
doc_inherit decorator

Usage:

class Foo(object):
    def foo(self):
        "Frobber"
        pass

class Bar(Foo):
    @doc_inherit
    def foo(self):
        pass

Now, Bar.foo.__doc__ == Bar().foo.__doc__ == Foo.foo.__doc__ == "Frobber"
"""

import warnings
from functools import wraps
import inspect

__all__ = ['doc_inherit']


class DocInherit(object):

    """
    Docstring inheriting method descriptor

    The class itself is also used as a decorator
    """

    def __init__(self, mthd):
        self.mthd = mthd
        self.name = mthd.__name__

    def __get__(self, obj, cls):
        if obj:
            return self.get_with_inst(obj, cls)
        else:
            return self.get_no_inst(cls)

    def get_with_inst(self, obj, cls):

        overridden = getattr(super(cls, obj), self.name, None)

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(obj, *args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def get_no_inst(self, cls):

        for parent in cls.__mro__[1:]:
            overridden = getattr(parent, self.name, None)
            if overridden:
                break

        @wraps(self.mthd, assigned=('__name__', '__module__'))
        def f(*args, **kwargs):
            return self.mthd(*args, **kwargs)

        return self.use_parent_doc(f, overridden)

    def use_parent_doc(self, func, source):
        if source is None:
            raise NameError("Can't find '%s' in parents" % self.name)
        func.__doc__ = source.__doc__
        return func

doc_inherit = DocInherit


def deprecated(msg):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Parameters
    ----------
    msg : str
        a user level hint which should indicate which feature to use otherwise.

    """
    def deprecated_decorator(func):

        def new_func(*args, **kwargs):
            _, filename, line_number, _, _, _ = \
                inspect.getouterframes(inspect.currentframe())[1]

            user_msg = "Call to deprecated function %s. Called from %s line %i. " \
                % (func.__name__, filename, line_number)
            if msg:
                user_msg += msg

            warnings.warn_explicit(
                user_msg,
                category=DeprecationWarning,
                filename=func.func_code.co_filename,
                lineno=func.func_code.co_firstlineno + 1
            )
            return func(*args, **kwargs)

        new_func.func_dict['__deprecated__'] = True

        # TODO: search docstring for notes section and append deprecation notice (with msg)

        return new_func

    return deprecated_decorator


def shortcut(name):
    """Add an shortcut (alias) to a decorated function.

    Calling the shortcut (alias) will call the decorated function. The shortcut name will be appended
    to the module's __all__ variable and the shortcut function will inherit the function's docstring

    Examples
    --------
    In some module you have defined a function
    >>> @shortcut('is_tmatrix') # doctest: +SKIP
    >>> def is_transition_matrix(args): # doctest: +SKIP
    >>>     pass # doctest: +SKIP
    Now you are able to call the function under its short name
    >>> is_tmatrix(args) # doctest: +SKIP

    """
    # TODO: this does not work (is not tested with class member functions)
    # extract callers frame
    frame = inspect.stack()[1][0]
    # get caller module of decorator

    def wrap(f):
        # docstrings are also being copied
        frame.f_globals[name] = f
        if frame.f_globals.has_key('__all__'):
            # add shortcut if it's not already there.
            if name not in frame.f_globals['__all__']:
                frame.f_globals['__all__'].append(name)
        return f
    return wrap
