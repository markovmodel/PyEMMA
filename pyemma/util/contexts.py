'''
Created on 04.01.2016

@author: marscher
'''

import random
import sys
from contextlib import contextmanager

import numpy as np


class conditional(object):
    """Wrap another context manager and enter it only if condition is true.
    """

    def __init__(self, condition, contextmanager):
        self.condition = condition
        self.contextmanager = contextmanager

    def __enter__(self):
        if self.condition:
            return self.contextmanager.__enter__()

    def __exit__(self, *args):
        if self.condition:
            return self.contextmanager.__exit__(*args)


@contextmanager
def numpy_random_seed(seed=42):
    """ sets the random seed of numpy within the context.

    Example
    -------
    >>> import numpy as np
    >>> with numpy_random_seed(seed=0):
    ...    np.random.randint(1000)
    684
    """
    old_state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(old_state)


@contextmanager
def random_seed(seed=42):
    """ sets the random seed of Python within the context.

    Example
    -------
    >>> import random
    >>> with random_seed(seed=0):
    ...    random.randint(0, 1000) # doctest: +SKIP
    864
    """
    old_state = random.getstate()
    random.seed(seed)
    try:
        yield
    finally:
        random.setstate(old_state)


@contextmanager
def settings(**kwargs):
    """ apply given PyEMMA config values temporarily within the given context."""
    from pyemma import config

    old_settings = {}
    try:
        # remember old setting, set new one. May raise ValueError, if invalid setting is given.
        for k, v in kwargs.items():
            old_settings[k] = getattr(config, k)
            setattr(config, k, v)
        yield
    finally:
        # restore old settings
        for k, v in old_settings.items():
            setattr(config, k, v)


@contextmanager
def named_temporary_file(mode='w+b', prefix='', suffix='', dir=None):
    from tempfile import NamedTemporaryFile
    ntf = NamedTemporaryFile(mode=mode, suffix=suffix, prefix=prefix, dir=dir, delete=False)
    ntf.close()
    try:
        yield ntf.name
    finally:
        import os
        try:
            os.unlink(ntf.name)
        except OSError:
            pass


class Capturing(list):
    """ captures specified stream lines in this wrapped list
    >>> with Capturing() as output:
    ...    print('hello world')
    >>> print(output)
    ['hello world']

    To capture stderr:
    >>> with Capturing(which='stderr') as output:
    ...    print('hello world', file=sys.stderr)
    >>> print(output)
    ['hello world']

    To extend the list, just pass it again:
    >>> with Capturing(append=output, which='stdout') as output:
    ...    print('hello again')
    >>> print(output)
    ['hello world', 'hello again']
    """
    def __init__(self, which='stdout', append=()):
        super(Capturing, self).__init__(append)
        self._which = which

    def __enter__(self):
        self._stream = getattr(sys, self._which)
        if sys.version_info[0] == 2:
            from StringIO import StringIO
        else:
            from io import StringIO
        self._stringio = StringIO()
        setattr(sys, self._which, self._stringio)
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        setattr(sys, self._which, self._stream)


class nullcontext(object):
    """Context manager that does no additional processing.
    Used as a stand-in for a normal context manager, when a particular
    block of code is only sometimes used with a normal context manager:
    cm = optional_cm if condition else nullcontext()
    with cm:
        # Perform operation, using optional_cm if condition is True
    """
    def __init__(self, enter_result=None):
        self.enter_result = enter_result
    def __enter__(self):
        return self.enter_result
    def __exit__(self, *excinfo):
        pass
