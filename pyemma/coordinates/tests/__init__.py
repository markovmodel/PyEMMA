
from __future__ import absolute_import


def setup_package():
    # setup function for nose tests (for this package only)
    from pyemma.util.config import conf_values
    # do not cache trajectory info in user directory (temp traj files)
    conf_values['pyemma']['use_trajectory_lengths_cache'] = False
