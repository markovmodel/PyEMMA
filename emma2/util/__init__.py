import numeric

def shortcut(alias_name, func):
    """
    creates a shortcut for given function 'func' under given alias name.
    
    Parameters
    ----------
    alias_name : name of alias
    func : function to create shortcut for

    Notes
    -----
    May be called only at module level!
    
    Example
    -------
    in module A
    >>> def foo():
    >>>     print 'hi'
    >>> bar = shortcut('bar', foo)
    >>> bar()
    hi
    """
    import sys
    dict_ = sys._getframe(1).f_globals
    f = func
    f.__doc__ = func.__doc__
    dict_['__all__'].append(alias_name) # append to all variable
    return f
