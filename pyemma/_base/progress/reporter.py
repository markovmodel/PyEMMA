
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

'''
Created on 16.07.2015

@author: marscher
'''
from __future__ import absolute_import, print_function
from pyemma.util.types import is_int
from pyemma._base.progress.bar import ProgressBar as _ProgressBar
from pyemma._base.progress.bar import show_progressbar as _show_progressbar
from pyemma._base.progress.bar.gui import hide_progressbar as _hide_progressbar


class ProgressReporter(object):
    """ Derive from this class to make some protected methods available to register
    and update status of different stages of an algorithm.
    """
    _pg_threshold = 2

    # Note: this class has intentionally no constructor, because it is more
    # comfortable for the user of this class (who is then not in the need to call it).

    def _progress_register(self, amount_of_work, description='', stage=0):
        """ Registers a progress which can be reported/displayed via a progress bar.

        Parameters
        ----------
        amount_of_work : int
            Amount of steps the underlying algorithm has to perform.
        description : str, optional
            This string will be displayed in the progress bar widget.
        stage : int, optional, default=0
            If the algorithm has multiple different stages (eg. calculate means
            in the first pass over the data, calculate covariances in the second),
            one needs to estimate different times of arrival.
        """
        if hasattr(self, 'show_progress') and not self.show_progress:
            return

        # note this semantic makes it possible to use this class without calling
        # its constructor.
        if not hasattr(self, '_prog_rep_progressbars'):
            self._prog_rep_progressbars = {}

        if not is_int(amount_of_work):
            raise ValueError("amount_of_work has to be of integer type. But is %s"
                             % type(amount_of_work))

        # if we do not have enough work to do for the overhead of a progress bar,
        # we just define a dummy here
        if amount_of_work <= ProgressReporter._pg_threshold:
            class dummy(object):
                pass
            pg = dummy()
            pg.__str__ = lambda: description
            pg.__repr__ = pg.__str__
            pg._dummy=None
            pg.description = ''
        else:
            pg = _ProgressBar(amount_of_work, description=description)

        self._prog_rep_progressbars[stage] = pg

#     def _progress_set_description(self, stage, description):
#         """ set description of an already existing progress """
#         assert hasattr(self, '_prog_rep_progressbars')
#         assert stage in self._prog_rep_progressbars
# 
#         self._prog_rep_progressbars[stage].description = description

    def register_progress_callback(self, call_back, stage=0):
        """ Registers the progress reporter.

        Parameters
        ----------
        call_back : function
            This function will be called with the following arguments:

            1. stage (int)
            2. instance of pyemma.utils.progressbar.ProgressBar
            3. optional \*args and named keywords (\*\*kw), for future changes

        stage: int, optional, default=0
            The stage you want the given call back function to be fired.
        """
        if hasattr(self, 'show_progress') and not self.show_progress:
            return

        if not hasattr(self, '_callbacks'):
            self._prog_rep_callbacks = {}

        assert callable(call_back)
        # check we have the desired function signature
        from pyemma.util.reflection import getargspec_no_self
        argspec = getargspec_no_self(call_back)
        assert len(argspec.args) == 2
        assert argspec.varargs is not None
        assert argspec.keywords is not None

        if stage not in self._prog_rep_callbacks:
            self._prog_rep_callbacks[stage] = []
        self._prog_rep_callbacks[stage].append(call_back)

    def _progress_update(self, numerator_increment, stage=0):
        """ Updates the progress. Will update progress bars or other progress output.

        Parameters
        ----------
        numerator : int
            numerator of partial work done already in current stage
        stage : int, nonnegative, default=0
            Current stage of the algorithm, 0 or greater

        """
        if hasattr(self, 'show_progress') and not self.show_progress:
            return

        if stage not in self._prog_rep_progressbars:
            raise RuntimeError(
                "call _progress_register(amount_of_work, stage=x) on this instance first!")
        if hasattr(self._prog_rep_progressbars[stage], '_dummy'):
            return

        pg = self._prog_rep_progressbars[stage]
        pg.numerator += numerator_increment

        if pg.numerator == pg.denominator:
            _hide_progressbar(pg)

        if pg.numerator > pg.denominator:
            raise Exception("This should not happen")

        _show_progressbar(pg)
        if hasattr(self, '_prog_rep_callbacks'):
            for callback in self._prog_rep_callbacks[stage]:
                callback(stage, pg)

    def _progress_force_finish(self, stage=0):
        """ forcefully finish the progress for given stage """
        if hasattr(self, 'show_progress') and not self.show_progress:
            return
        if stage not in self._prog_rep_progressbars:
            raise RuntimeError(
                "call _progress_register(amount_of_work, stage=x) on this instance first!")
        pg = self._prog_rep_progressbars[stage]
        if not isinstance(pg, _ProgressBar):
            return
        pg.numerator = pg.denominator
        pg._eta.eta_epoch = 0
        _show_progressbar(pg)
        _hide_progressbar(pg)
