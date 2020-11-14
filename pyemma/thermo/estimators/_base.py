
class Undefined(object): pass


class ThermoBase(object):

    __serialize_version = 0
    __serialize_fields = ('umbrella_centers', 'force_constants', 'temperatures', 'dt_traj')

    @property
    def umbrella_centers(self):
        """ The individual umbrella centers labelled accordingly to ttrajs.
        (only set, when estimated from umbrella data).
        """
        try:
            return self._umbrella_centers
        except AttributeError:
            return Undefined

    @umbrella_centers.setter
    def umbrella_centers(self, value):
        self._umbrella_centers = value

    @property
    def force_constants(self):
        """The individual force matrices labelled accordingly to ttrajs.
        (only set, when estimated from umbrella data).
        """
        try:
            return self._force_constants
        except AttributeError:
            return Undefined

    @force_constants.setter
    def force_constants(self, value):
        self._force_constants = value

    @property
    def temperatures(self):
        """ The individual temperatures labelled accordingly to ttrajs.
        (only set, when estimated from multi-temperature data).
        """
        try:
            return self._temperatures
        except AttributeError:
            return Undefined

    @temperatures.setter
    def temperatures(self, value):
        self._temperatures = value

    @property
    def dt_traj(self):
        return self._dt_traj

    @dt_traj.setter
    def dt_traj(self, value):
        # time step
        self._dt_traj = value
        from pyemma.util.units import TimeUnit
        self.timestep_traj = TimeUnit(self.dt_traj)
