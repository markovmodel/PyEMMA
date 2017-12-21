
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

from pyemma._base.serialization.serialization import SerializableMixIn

import numpy as np
import math

__author__ = 'noe'


class TimeUnit(SerializableMixIn):
    __serialize_version = 0
    __serialize_fields = ('_unit', '_factor')

    _UNIT_STEP = -1
    _UNIT_FS = 0
    _UNIT_PS = 1
    _UNIT_NS = 2
    _UNIT_US = 3
    _UNIT_MS = 4
    _UNIT_S  = 5
    _unit_names = ['fs','ps','ns','us','ms','s']

    def __init__(self, unit = '1 step'):
        """
        Initializes the time unit object

        Parameters
        ----------
        unit : str
            Description of a physical time unit. By default '1 step', i.e. there is no physical time unit.
            Specify by a number, whitespace and unit. Permitted units are (* is an arbitrary string):
            'fs',  'femtosecond*'
            'ps',  'picosecond*'
            'ns',  'nanosecond*'
            'us',  'microsecond*'
            'ms',  'millisecond*'
            's',   'second*'

        """
        if isinstance(unit, TimeUnit):  # copy constructor
            self._factor = unit._factor
            self._unit = unit._unit
        elif isinstance(unit, str):  # construct from string
            lunit = unit.lower()
            words = lunit.split(' ')

            if len(words) == 1:
                self._factor = 1.0
                unitstring = words[0]
            elif len(words) == 2:
                self._factor = float(words[0])
                unitstring = words[1]
            else:
                raise ValueError('Illegal input string: '+str(unit))

            if unitstring == 'step':
                self._unit = self._UNIT_STEP
            elif unitstring == 'fs' or unitstring.startswith('femtosecond'):
                self._unit = self._UNIT_FS
            elif unitstring == 'ps' or unitstring.startswith('picosecond'):
                self._unit = self._UNIT_PS
            elif unitstring == 'ns' or unitstring.startswith('nanosecond'):
                self._unit = self._UNIT_NS
            elif unitstring == 'us' or unitstring.startswith('microsecond'):
                self._unit = self._UNIT_US
            elif unitstring == 'ms' or unitstring.startswith('millisecond'):
                self._unit = self._UNIT_MS
            elif unitstring == 's' or unitstring.startswith('second'):
                self._unit = self._UNIT_S
            else:
                raise ValueError('Time unit is not understood: '+unit)
        else:
            raise ValueError('Unknown type: %s' % unit)

    def __str__(self):
        if not hasattr(self, '_unit'):
            return "[TimeUnit {unknown}]"

        if self._unit == -1:
            return str(self._factor)+' step'
        else:
            return str(self._factor)+' '+self._unit_names[self._unit]

    def __repr__(self):
        return "[TimeUnit {}]".format(self)

    def __eq__(self, other):
        return isinstance(other, TimeUnit) and self.dt == other.dt and self.unit == other.unit

    @property
    def dt(self):
        return self._factor

    @property
    def unit(self):
        return self._unit

    def get_scaled(self, factor):
        """ Get a new time unit, scaled by the given factor """
        res = self.__new__(self.__class__)
        res._factor = self._factor * factor
        res._unit = self._unit
        return res

    def rescale_around1(self, times):
        """
        Suggests a rescaling factor and new physical time unit to balance the given time multiples around 1.

        Parameters
        ----------
        times : float array
            array of times in multiple of the present elementary unit

        """
        if self._unit == self._UNIT_STEP:
            return times, 'step' # nothing to do

        m = np.mean(times)
        mult = 1.0
        cur_unit = self._unit

        # numbers are too small. Making them larger and reducing the unit:
        if (m < 0.001):
            while mult*m < 0.001 and cur_unit >= 0:
                mult *= 1000
                cur_unit -= 1
            return mult*times, self._unit_names[cur_unit]

        # numbers are too large. Making them smaller and increasing the unit:
        if (m > 1000):
            while mult*m > 1000 and cur_unit <= 5:
                mult /= 1000
                cur_unit += 1
            return mult*times, self._unit_names[cur_unit]

        # nothing to do
        return times, self._unit

def bytes_to_string(num, suffix='B'):
    """
    Returns the size of num (bytes) in a human readable form up to Yottabytes (YB).
    :param num: The size of interest in bytes.
    :param suffix: A suffix, default 'B' for 'bytes'.
    :return: a human readable representation of a size in bytes
    """
    extensions = ["%s%s" % (x, suffix) for x in ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']]
    if num == 0:
        return "0%s" % extensions[0]
    else:
        n_bytes = float(abs(num))
        place = int(math.floor(math.log(n_bytes, 1024)))
        return "%.1f%s" % (np.sign(num) * (n_bytes / 1024** place), extensions[place])


def string_to_bytes(string):
    """
    Returns the amount of bytes in a human readable form up to Yottabytes (YB).
    :param string: integer with suffix (b, k, m, g, t, p, e, z, y)
    :return: amount of bytes in string representation

    >>> string_to_bytes('1024')
    1024
    >>> string_to_bytes('1024k')
    1048576
    >>> string_to_bytes('4 G')
    4294967296
    >>> string_to_bytes('4.5g')
    4831838208
    >>> try:
    ...     string_to_bytes('1x')
    ... except RuntimeError as re:
    ...     assert 'unknown suffix' in str(re)
    """
    if string == '0':
        return 0
    import re
    match = re.match('(\d+\.?\d?)\s?([bBkKmMgGtTpPeEzZyY])?(\D?)', string)
    if not match:
        raise RuntimeError('"{}" does not match "[integer] [suffix]"'.format(string))
    if match.group(3):
        raise RuntimeError('unknown suffix: "{}"'.format(match.group(3)))
    value = float(match.group(1))
    if match.group(2) is None:
        return int(value)
    suffix = match.group(2).upper()
    extensions = ['', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y']
    x = extensions.index(suffix)
    value *= 1024**x
    return int(value)
