
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
Created on 29.07.2015

@author: marscher
'''

from __future__ import absolute_import
import unittest
from pyemma._base.progress import ProgressReporter
from pyemma._base.progress.bar import ProgressBar
from six.moves import range


class TestProgress(unittest.TestCase):

    # FIXME: does not work with nose (because nose already captures stdout)
    """
    def test_silenced(self):
        reporter = ProgressReporter()
        reporter._register(1)
        reporter.silence_progress = True

        from StringIO import StringIO

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            # call the update method to potentially create output
            reporter._update(1)
            output = out.getvalue().strip()
            # in silence mode we do not want any output!
            assert output == ''
        finally:
            sys.stdout = saved_stdout

    def test_not_silenced(self):
        reporter = ProgressReporter()
        reporter._register(1)
        reporter.silence_progress = False

        from StringIO import StringIO

        saved_stdout = sys.stdout
        try:
            out = StringIO()
            sys.stdout = out
            # call the update method to potentially create output
            reporter._update(1)
            output = out.getvalue().strip()
            # in silence mode we do not want any output!
            print output
            assert output is not ''
        finally:
            sys.stdout = saved_stdout
    """

    def test_callback(self):

        self.has_been_called = 0

        def call_back(stage, progressbar, *args, **kw):
            self.has_been_called += 1
            assert isinstance(stage, int)
            assert isinstance(progressbar, ProgressBar)

        amount_of_work = 100
        worker = ProgressReporter()
        worker._progress_register(
            amount_of_work, description="hard working", stage=0)
        worker.register_progress_callback(call_back, stage=0)
        for _ in range(amount_of_work):
            worker._progress_update(1, stage=0)
        self.assertEqual(self.has_been_called, amount_of_work)

if __name__ == "__main__":
    unittest.main()
