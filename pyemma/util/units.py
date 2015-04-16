
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__author__ = 'noe'

import numpy as np

class TimeUnit:

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

    def __str__(self):
        if self._unit == -1:
            return str(self._factor)+' step'
        else:
            return str(self._factor)+' '+self._unit_names[self._unit]

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