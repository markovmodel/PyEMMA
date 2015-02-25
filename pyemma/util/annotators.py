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


def shortcut(name):
    """ add an shortcut (alias) to a decorated function.

    The alias function will have the same docstring and will be appended to
    the module __all__ variable, where the original function is defined.

    Examples
    --------
    In some module you have defined a function
    >>>@shortcut('is_tmatrix')
    >>>def is_transition_matrix(args):
    >>>    pass
    Now you are able to call the function under its short name
    >>> is_tmatrix(args)

    """
    # extract callers frame
    frame = inspect.stack()[1][0]
    # get caller module of decorator

    def wrap(f):
        # docstrings are also being copied
        frame.f_globals[name] = f
        if frame.f_globals.has_key('__all__'):
            frame.f_globals['__all__'].append(name)
        return f
    return wrap
