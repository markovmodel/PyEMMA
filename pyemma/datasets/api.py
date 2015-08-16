__author__ = 'noe'

def load_2well_discrete():
    from double_well_discrete import DoubleWell_Discrete_Data
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
    trajs = sorted(filter(lambda x: x.find('mini') == -1, glob(path + os.sep + "*.xtc")))
    return {'trajs': trajs, 'top': top}
