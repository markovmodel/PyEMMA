from __future__ import (print_function, absolute_import)
__author__ = 'noe'

def load_2well_discrete():
    from .double_well_discrete import DoubleWell_Discrete_Data
    return DoubleWell_Discrete_Data()

def get_bpti_test_data():
    """ Returns a dictionary containing C-alpha coordinates of a truncated
    BTBI trajectory.

    Notes
    -----
    You will have to load the data from disc yourself. See eg.
    :py:func:`pyemma.coordinates.load`.

    Returns
    -------
    res : {trajs : list, top: str}
        trajs is a list of filenames
        top is str pointing to the path of the topology file.
    """
    import os
    from glob import glob
    import pkg_resources
    path = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/')
    top = pkg_resources.resource_filename('pyemma.coordinates.tests', 'data/bpti_ca.pdb')

    trajs = glob(path + os.sep + "*.xtc")
    trajs = filter(lambda f: not f.endswith("bpti_mini.xtc"), trajs)
    trajs = sorted(trajs)

    assert len(trajs) == 3
    return {'trajs': trajs, 'top': top}
