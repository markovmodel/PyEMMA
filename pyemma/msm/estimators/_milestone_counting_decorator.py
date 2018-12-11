
# This file is part of PyEMMA.
#
# Copyright (c) 2017 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
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
from decorator import decorator

from pyemma.util.discrete_trajectories import milestone_counting

""" attribute name of dynamically added attribute for index offsets."""
_OFFSETS_ATTR_NAME = 'dtrajs_milestone_counting_offsets'
__author__ = 'marscher'


def _MilestoneCountingDecorator(cls):
    """ overrides the _estimate method of cls to perform milestone counting """
    assert hasattr(cls, '_estimate')
    old_estimate = cls._estimate

    @property
    def dtrajs_milestone_counting_offsets(self):
        """ Offsets for milestoned trajectories for each input discrete trajectory """
        return self._dtrajs_milestone_counting_offsets

    @dtrajs_milestone_counting_offsets.setter
    def dtrajs_milestone_counting_offsets(self, value):
        self._dtrajs_milestone_counting_offsets = value

    def _estimate(self, dtrajs):
        assert hasattr(self, 'core_set')

        dtrajs_core, offsets, n_cores = milestone_counting(dtrajs, core_set=self.core_set, in_place=False)
        setattr(self, _OFFSETS_ATTR_NAME, offsets)
        self.n_cores = n_cores
        # do the actual estimation
        return old_estimate(self, dtrajs_core)

    cls._estimate = _estimate
    cls.dtrajs_milestone_counting_offsets = dtrajs_milestone_counting_offsets
    return cls


@decorator
def _remap_indices_coring(func, self, *args, **kwargs):
    """Since milestone counting sometimes has to truncate the discrete trajectories (eg. outliers),
    it becomes mission crucial to maintain the mapping to of the indices to the original input trajectories.
    """
    indices = func(self, *args, **kwargs)
    if hasattr(self, _OFFSETS_ATTR_NAME) and any(getattr(self, _OFFSETS_ATTR_NAME)):  # need to remap indices?
        import numpy as np
        from pyemma.util.discrete_trajectories import _apply_offsets_to_samples
        dtraj_offsets  = getattr(self, _OFFSETS_ATTR_NAME)
        if isinstance(indices, np.ndarray) and indices.dtype == np.int_:
            _apply_offsets_to_samples(indices, dtraj_offsets)
        elif isinstance(indices, list) or (isinstance(indices, np.ndarray) and indices.dtype == np.object_):
            for s in indices:
                _apply_offsets_to_samples(s, dtraj_offsets)

    return indices
