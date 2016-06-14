"""A shim module for deprecated imports
"""
# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import sys
import types
import warnings


def import_item(name):
    """Import and return ``bar`` given the string ``foo.bar``.
    Calling ``bar = import_item("foo.bar")`` is the functional equivalent of
    executing the code ``from foo import bar``.
    Parameters
    ----------
    name : string
      The fully qualified name of the module/package being imported.
    Returns
    -------
    mod : module object
       The module that was imported.
    """

    parts = name.rsplit('.', 1)
    if len(parts) == 2:
        # called with 'foo.bar....'
        package, obj = parts
        module = __import__(package, fromlist=[obj])
        try:
            pak = getattr(module, obj)
        except AttributeError:
            raise ImportError('No module named %s' % obj)
        return pak
    else:
        # called with un-dotted string
        return __import__(parts[0])


class ShimImporter(object):
    """Import hook for a shim.

    This ensures that submodule imports return the real target module,
    not a clone that will confuse `is` and `isinstance` checks.
    """

    def __init__(self, src, mirror):
        self.src = src
        self.mirror = mirror

    def _mirror_name(self, fullname):
        """get the name of the mirrored module"""

        return self.mirror + fullname[len(self.src):]

    def find_module(self, fullname, path=None):
        """Return self if we should be used to import the module."""
        if fullname.startswith(self.src + '.'):
            mirror_name = self._mirror_name(fullname)
            try:
                mod = import_item(mirror_name)
            except ImportError:
                return
            else:
                if not isinstance(mod, types.ModuleType):
                    # not a module
                    return None
                return self

    def load_module(self, fullname):
        """Import the mirrored module, and insert it into sys.modules"""
        mirror_name = self._mirror_name(fullname)
        mod = import_item(mirror_name)
        sys.modules[fullname] = mod
        return mod


class ShimModule(types.ModuleType):
    def __init__(self, *args, **kwargs):
        self._mirror = kwargs.pop("mirror")
        self._from = self._mirror.split('.')
        src = kwargs.pop("src", None)
        if src:
            kwargs['name'] = src.rsplit('.', 1)[-1]
        super(ShimModule, self).__init__(*args, **kwargs)
        # add import hook for descendant modules
        if src:
            sys.meta_path.append(
                ShimImporter(src=src, mirror=self._mirror)
            )
        self.msg = kwargs.pop("msg", None)
        self.default_msg = "Access to a moved module '%s' detected!" \
                           " Please use '%s' in the future." % (src, self._mirror)

    @property
    def __path__(self):
        return []

    @property
    def __spec__(self):
        """Don't produce __spec__ until requested"""
        self._warn('__spec__')
        return import_item(self._mirror).__spec__

    def __dir__(self):
        self._warn('__dir__')
        return dir(import_item(self._mirror))

    @property
    def __all__(self):
        """Ensure __all__ is always defined"""
        self._warn('__all__')
        mod = import_item(self._mirror)
        try:
            return mod.__all__
        except AttributeError:
            return [name for name in dir(mod) if not name.startswith('_')]

    def __getattr__(self, key):
        # Use the equivalent of import_item(name), see below
        name = "%s.%s" % (self._mirror, key)
        try:
            item = import_item(name)
            self._warn('__getattr__')
            return item
        except ImportError:
            raise AttributeError(key)

    def _warn(self, called_from):
        from pyemma.util.exceptions import PyEMMA_DeprecationWarning
        warnings.warn(self.msg if self.msg else self.default_msg,
                      category=PyEMMA_DeprecationWarning)
