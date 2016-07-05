# This file is part of PyEMMA.
#
# Copyright (c) 2016 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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

import numpy as _np

class SubSet(object):
    @property
    def active_set(self):
        if not hasattr(self, "_active_set"):
            self._active_set = []
        return self._active_set
    @active_set.setter
    def active_set(self, active_set):
        self._active_set = _np.asarray(active_set, dtype=_np.intc)
    @property
    def nstates_full(self):
        if not hasattr(self, "_nstates_full"):
            self._nstates_full = len(self.active_set)
        return self._nstates_full
    @nstates_full.setter
    def nstates_full(self, nstates_full):
        self._nstates_full = int(nstates_full)
