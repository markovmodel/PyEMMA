
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
import warnings

from decorator import decorator, decorate
from inspect import stack

from pyemma.util.exceptions import PyEMMA_DeprecationWarning

__all__ = ['alias',
           'aliased',
           'deprecated',
           'shortcut',
           'fix_docs',
           ]


def fix_docs(cls):
    """ copies docstrings of derived attributes (methods, properties, attrs) from parent classes."""
    import inspect
    public_undocumented_members = {name: func for name, func in inspect.getmembers(cls)
                                   if not name.startswith('_') and func.__doc__ is None}

    for name, func in public_undocumented_members.items():
        for parent in cls.__mro__[1:]:
            parfunc = getattr(parent, name, None)
            if parfunc and getattr(parfunc, '__doc__', None):
                if isinstance(func, property):
                    # copy property, since its doc attribute is read-only
                    new_prop = property(fget=func.fget, fset=func.fset,
                                        fdel=func.fdel, doc=parfunc.__doc__)
                    setattr(cls, name, new_prop)
                else:
                    if hasattr(func, '__func__'):  # handle instancemethods
                        func.__func__.__doc__ = parfunc.__doc__
                    else:
                        func.__doc__ = parfunc.__doc__
                break
    return cls


class alias(object):
    """
    Alias class that can be used as a decorator for making methods callable
    through other names (or "aliases").
    Note: This decorator must be used inside an @aliased -decorated class.
    For example, if you want to make the method shout() be also callable as
    yell() and scream(), you can use alias like this:

        @alias('yell', 'scream')
        def shout(message):
            # ....
    """

    def __init__(self, *aliases):
        """Constructor."""
        self.aliases = set(aliases)

    def __call__(self, f):
        """
        Method call wrapper. As this decorator has arguments, this method will
        only be called once as a part of the decoration process, receiving only
        one argument: the decorated function ('f'). As a result of this kind of
        decorator, this method must return the callable that will wrap the
        decorated function.
        """
        if isinstance(f, property):
            f.fget._aliases = self.aliases
        else:
            f._aliases = self.aliases
        return f


def aliased(aliased_class):
    """
    Decorator function that *must* be used in combination with @alias
    decorator. This class will make the magic happen!
    @aliased classes will have their aliased method (via @alias) actually
    aliased.
    This method simply iterates over the member attributes of 'aliased_class'
    seeking for those which have an '_aliases' attribute and then defines new
    members in the class using those aliases as mere pointer functions to the
    original ones.

    Usage:

    >>> @aliased
    ... class MyClass(object):
    ...     @alias('coolMethod', 'myKinkyMethod')
    ...     def boring_method(self):
    ...        pass
    ...
    ...     @property
    ...     @alias('my_prop_alias')
    ...     def my_prop(self):
    ...        return "hi"

    >>> i = MyClass()
    >>> i.coolMethod() # equivalent to i.myKinkyMethod() and i.boring_method()
    >>> i.my_prop == i.my_prop_alias
    True

    """
    original_methods = aliased_class.__dict__.copy()
    original_methods_set = set(original_methods)
    for name, method in original_methods.items():
        aliases = None
        if isinstance(method, property) and hasattr(method.fget, '_aliases'):
            aliases = method.fget._aliases
        elif hasattr(method, '_aliases'):
            aliases = method._aliases

        if aliases:
            # Add the aliases for 'method', but don't override any
            # previously-defined attribute of 'aliased_class'
            for alias in aliases - original_methods_set:
                setattr(aliased_class, alias, method)
    return aliased_class


def shortcut(*names):
    """Add an shortcut (alias) to a decorated function, but not to class methods!

    Use aliased/alias decorators for class members!

    Calling the shortcut (alias) will call the decorated function. The shortcut name will be appended
    to the module's __all__ variable and the shortcut function will inherit the function's docstring

    Examples
    --------
    In some module you have defined a function
    >>> @shortcut('is_tmatrix') # doctest: +SKIP
    >>> def is_transition_matrix(args): # doctest: +SKIP
    ...     pass # doctest: +SKIP
    Now you are able to call the function under its short name
    >>> is_tmatrix(args) # doctest: +SKIP

    """
    def wrap(f):
        globals_ = f.__globals__
        for name in names:
            globals_[name] = f
            if '__all__' in globals_ and name not in globals_['__all__']:
                globals_['__all__'].append(name)
        return f
    return wrap


def get_culprit(omit_top_frames=1):
    """get the filename and line number calling this.

    Parameters
    ----------
    omit_top_frames: int, default=1
        omit n frames from top of stack stack. Purpose is to get the real
        culprit and not intermediate functions on the stack.
    Returns
    -------
    (filename: str, fileno: int)
    filename and line number of the culprit.
    """
    try:
        caller_stack = stack()[omit_top_frames:]
        while len(caller_stack) > 0:
            frame = caller_stack.pop(0)
            filename = frame[1]
            # skip callee frames if they are other decorators or this file(func)
            if '<decorator' in filename or __file__ in filename:
                continue
            else:
                break
        lineno = frame[2]
        # avoid cyclic references!
        del caller_stack, frame
    except OSError:  # eg. os.getcwd() fails in conda-test, since cwd gets deleted.
        filename = 'unknown'
        lineno = -1
    return filename, lineno


def deprecated(*optional_message):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.

    Parameters
    ----------
    *optional_message : str
        an optional user level hint which should indicate which feature to use otherwise.

    """
    def _deprecated(func, *args, **kw):
        filename, lineno = get_culprit()
        user_msg = 'Call to deprecated function "%s". Called from %s line %i. %s' \
                   % (func.__name__, filename, lineno, msg)

        warnings.warn_explicit(
            user_msg,
            category=PyEMMA_DeprecationWarning,
            filename=filename,
            lineno=lineno
        )
        return func(*args, **kw)

    # add deprecation notice to func docstring:
    if len(optional_message) == 1 and callable(optional_message[0]):
        # this is the function itself, decorate!
        msg = ""
        return decorate(optional_message[0], _deprecated)
    else:
        # actually got a message (or empty parenthesis)
        msg = optional_message[0] if len(optional_message) > 0 else ""
        return decorator(_deprecated)
