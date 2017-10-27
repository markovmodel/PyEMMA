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

import time


class _ProgressIndicatorCallBack(object):
    def __init__(self):
        self.time = 0.0

    # TODO: unify this concept in ProgressReporter (but make it adaptive)
    def waiting(self):
        now = time.time()
        if now - self.time < .2:
            return True
        else:
            self.time = now
            return False


class _IterationProgressIndicatorCallBack(_ProgressIndicatorCallBack):
    def __init__(self, reporter, description, stage):
        super(_IterationProgressIndicatorCallBack, self).__init__()
        reporter._progress_register(10, description, stage=stage, tqdm_args=dict(smoothing=0.5))
        self.stage = stage
        self.reporter = reporter

    def __call__(self, *args, **kwargs):
        if not self.reporter.show_progress: return
        if self.waiting(): return
        self.reporter._prog_rep_progressbars[self.stage].total = kwargs['maxiter']
        self.reporter._progress_update(1, stage=self.stage)


class _ConvergenceProgressIndicatorCallBack(_ProgressIndicatorCallBack):
    def __init__(self, reporter, stage, maxiter, maxerr, subcallback=None):
        self.template = str(stage) + ' increment={err:0.1e}/{maxerr:0.1e}'
        description = str(stage)
        super(_ConvergenceProgressIndicatorCallBack, self).__init__()
        self.final = maxiter
        reporter._progress_register(int(self.final), description, stage=stage, tqdm_args=dict(smoothing=0.5))
        self.stage = stage
        self.reporter = reporter
        self.state = 0.0
        self.subcallback = subcallback

    def __call__(self, *args, **kwargs):
        if self.subcallback is not None:
            self.subcallback(*args, **kwargs)
        if not self.reporter.show_progress: return
        if self.waiting(): return
        current = kwargs['iteration_step']
        if 'err' in kwargs and 'maxerr' in kwargs:
            err_str = self.template.format(err=kwargs['err'], maxerr=kwargs['maxerr'])
            self.reporter._progress_set_description(self.stage, err_str)
        if current > 0.0 and self.state < current <= self.final:
            difference = current - self.state
            self.reporter._progress_update(difference, stage=self.stage)
            self.state = current
