from __future__ import absolute_import
from six.moves import range
__author__ = 'noe'

import numpy as np
from pyemma.msm import markov_model

class DoubleWell_Discrete_Data(object):
    """ MCMC process in a symmetric double well potential, spatially discretized to 100 bins

    """

    def __init__(self):
        from pkg_resources import resource_filename
        filename = resource_filename('pyemma.datasets', 'double_well_discrete.npz')
        datafile = np.load(filename)
        self._dtraj_T100K_dt10 = datafile['dtraj']
        self._P = datafile['P']
        self._msm = markov_model(self._P)

    @property
    def dtraj_T100K_dt10(self):
        """ 100K frames trajectory at timestep 10, 100 microstates (not all are populated). """
        return self._dtraj_T100K_dt10

    @property
    def dtraj_T100K_dt10_n2good(self):
        """ 100K frames trajectory at timestep 10, good 2-state discretization (at transition state). """
        return self.dtraj_T100K_dt10_n([50])

    @property
    def dtraj_T100K_dt10_n2bad(self):
        """ 100K frames trajectory at timestep 10, bad 2-state discretization (off transition state). """
        return self.dtraj_T100K_dt10_n([40])

    def dtraj_T100K_dt10_n2(self, divide):
        """ 100K frames trajectory at timestep 10, arbitrary 2-state discretization. """
        return self.dtraj_T100K_dt10_n([divide])

    @property
    def dtraj_T100K_dt10_n6good(self):
        """ 100K frames trajectory at timestep 10, good 6-state discretization. """
        return self.dtraj_T100K_dt10_n([40, 45, 50, 55, 60])

    def dtraj_T100K_dt10_n(self, divides):
        """ 100K frames trajectory at timestep 10, arbitrary n-state discretization. """
        disc = np.zeros(100, dtype=int)
        divides = np.concatenate([divides, [100]])
        for i in range(len(divides)-1):
            disc[divides[i]:divides[i+1]] = i+1
        return disc[self.dtraj_T100K_dt10]

    @property
    def transition_matrix(self):
        """ Exact transition matrix used to generate the data """
        return self._P

    @property
    def msm(self):
        """ Returns an MSM object with the exact transition matrix """
        return self._msm

    def generate_traj(self, N, start=None, stop=None, dt=1):
        """ Generates a random trajectory of length N with time step dt """
        from msmtools.generation import generate_traj
        return generate_traj(self._P, N, start=start, stop=stop, dt=dt)

    def generate_trajs(self, M, N, start=None, stop=None, dt=1):
        """ Generates M random trajectories of length N each with time step dt """
        from msmtools.generation import generate_trajs
        return generate_trajs(self._P, M, N, start=start, stop=stop, dt=dt)
