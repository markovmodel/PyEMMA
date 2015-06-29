__author__ = 'noe'


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
    if not isinstance(f, str):
        fname = f.im_func.func_name
    else:
        fname = f
    # get the method ref
    method = getattr(obj, fname)
    # handle cases
    if str(type(method)) == '<type \'instancemethod\'>':  # call function without params
        return method(*args, **kwargs)
    elif str(type(method)) == '<type \'property\'>':  # call property
        return method
    else:  # now it's an Attribute, so we can just return its value
        return method
