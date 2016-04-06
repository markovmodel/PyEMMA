
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

"""Easy to use ETA calculation. Used by etaprogress.progress.
taken from:

https://github.com/Robpol86/etaprogress
https://pypi.python.org/pypi/etaprogress
"""

from __future__ import absolute_import

from __future__ import division
from collections import deque
from math import sqrt, ceil
import time
import datetime
import locale
from itertools import cycle
from .misc import terminal_width

SPINNER = cycle(('/', '-', '\\', '|'))

__all__ = ('ProgressBar',
           )
_NOW = time.time  # For testing.


class ETA(object):

    """Calculates the estimated time remaining using Simple Linear Regression.

    If `denominator` is 0 or None, no ETA will be available.

    Keyword arguments:
    denominator -- the final/total number of units (like the expected file size of a download). 0 if unknown.
    scope -- used up to these many recent numerator entries to calculate the rate and ETA. Default is 60.

    Instance variables:
    eta_epoch -- expected time (seconds since Unix epoch or time.time()) of completion (float).
    rate -- current rate of progress (float).
    _start_time -- used to derive the amount of time everything took to reach 100%.
    _timing_data -- deque instance holding timing data. Similar to a list. Items are 2-item tuples, first item (x) is
        time.time(), second item (y) is the numerator. Limited to `scope` to base ETA on. Once this limit is reached,
        any new numerator item pushes off the oldest entry from the deque instance.
    """

    def __init__(self, denominator=0, scope=60):
        self.denominator = denominator
        self.eta_epoch = None
        self.rate = 0.0

        self._start_time = _NOW()
        self._timing_data = deque(maxlen=scope)

    @property
    def numerator(self):
        """Returns the latest numerator."""
        return self._timing_data[-1][1] if self._timing_data else 0

    @numerator.setter
    def numerator(self, value):
        """Sets a new numerator (adds to timing data table). Must be greater than or equal to previous numerator."""
        self.set_numerator(value)

    @property
    def stalled(self):
        """Returns True if the rate is 0."""
        return float(self.rate) == 0.0

    @property
    def started(self):
        """Returns True if there is enough data to calculate the rate."""
        return len(self._timing_data) >= 2

    @property
    def undefined(self):
        """Returns True if there is no denominator."""
        return self.denominator is None or self.denominator <= 0

    @property
    def done(self):
        """Returns True if numerator == denominator."""
        return False if self.undefined else self.numerator == self.denominator

    @property
    def eta_seconds(self):
        """Returns the ETA in seconds or None if there is no data yet."""
        return None if self.eta_epoch is None else max([self.eta_epoch - _NOW(), 0])

    @property
    def eta_timediff(self):
        """ Returns timediff of seconds left """
        return None if self.eta_epoch is None else datetime.timedelta(
            seconds=max([self.eta_epoch - _NOW(), 0]))

    @property
    def percent(self):
        """Returns the percent as a float."""
        return 0.0 if self.undefined else self.numerator / self.denominator * 100

    @property
    def elapsed(self):
        """Returns the number of seconds it has been since the start until the latest entry."""
        if not self.started or self._start_time is None:
            return 0.0
        return self._timing_data[-1][0] - self._start_time

    @property
    def rate_unstable(self):
        """Returns an unstable rate based on the last two entries in the timing data. Less intensive to compute."""
        if not self.started or self.stalled:
            return 0.0
        x1, y1 = self._timing_data[-2]
        x2, y2 = self._timing_data[-1]
        return (y2 - y1) / (x2 - x1)

    @property
    def rate_overall(self):
        """Returns the overall average rate based on the start time."""
        elapsed = self.elapsed
        return self.rate if not elapsed else self.numerator / self.elapsed

    def set_numerator(self, numerator, calculate=True):
        """Sets the new numerator (number of items done).

        Positional arguments:
        numerator -- the new numerator to add to the timing data.

        Keyword arguments:
        calculate -- calculate the ETA and rate by default.
        """
        # Validate
        if self._timing_data and numerator < self._timing_data[-1][1]:
            raise ValueError('numerator cannot decrement.')

        # Update data.
        now = _NOW()
        if self._timing_data and now == self._timing_data[-1][0]:
            self._timing_data[-1] = (now, numerator)  # Overwrite.
        else:
            self._timing_data.append((now, numerator))

        # Calculate ETA and rate.
        if not self.done and calculate and self.started:
            self._calculate()

    def _calculate(self):
        """Perform the ETA and rate calculation.

        Two linear lines are used to calculate the ETA: the linear regression (line through a scatter-plot), and the
        fitted line (a line that runs through the latest data point in _timing_data but parallel to the linear
        regression line).

        As the percentage moves closer to 100%, _calculate() gradually uses the ETA based on the fitted line more and
        more. This is done to prevent an ETA that's in the past.

        http://code.activestate.com/recipes/578914-simple-linear-regression-with-pure-python/
        http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient
        """
        # Calculate means and standard deviations.
        mean_x = sum(i[0] for i in self._timing_data) / len(self._timing_data)
        mean_y = sum(i[1] for i in self._timing_data) / len(self._timing_data)
        std_x = sqrt(sum(pow(i[0] - mean_x, 2)
                         for i in self._timing_data) / (len(self._timing_data) - 1))
        std_y = sqrt(sum(pow(i[1] - mean_y, 2)
                         for i in self._timing_data) / (len(self._timing_data) - 1))

        # Calculate coefficient.
        sum_xy, sum_sq_v_x, sum_sq_v_y = 0, 0, 0
        for x, y in self._timing_data:
            x -= mean_x
            y -= mean_y
            sum_xy += x * y
            sum_sq_v_x += pow(x, 2)
            sum_sq_v_y += pow(y, 2)
        pearson_r = sum_xy / sqrt(sum_sq_v_x * sum_sq_v_y)

        # Calculate regression line. y = mx + b where m is the slope and b is
        # the y-intercept.
        m = self.rate = pearson_r * (std_y / std_x)
        if self.undefined:
            return
        y = self.denominator
        b = mean_y - m * mean_x
        x = (y - b) / m

        # Calculate fitted line (transformed/shifted regression line
        # horizontally).
        fitted_b = self._timing_data[-1][1] - (m * self._timing_data[-1][0])
        fitted_x = (y - fitted_b) / m
        adjusted_x = ((fitted_x - x) * (self.numerator / self.denominator)) + x
        self.eta_epoch = adjusted_x


class BaseProgressBar(object):

    """Holds common properties/methods/etc for ProgressBar and related subclasses."""

    def __init__(self, denominator, max_width=None, eta_every=1):
        self._eta = ETA(denominator=denominator)
        self.max_width = max_width
        self.eta_every = eta_every
        self.force_done = False
        self._eta_string = ''
        self._eta_count = 1

    @staticmethod
    def _generate_eta(seconds):
        """Kind of like an interface method, to be implemented by subclasses."""
        raise NotImplementedError

    @property
    def denominator(self):
        """Returns the denominator as an integer."""
        return int(self._eta.denominator)

    @denominator.setter
    def denominator(self, val):
        val = int(val)
        self._eta.denominator = int(val)

    @property
    def done(self):
        """Returns True if the progress has completed."""
        if self.force_done:
            return True
        return self._eta.done

    @property
    def numerator(self):
        """Returns the numerator as an integer."""
        return self._eta.numerator

    @numerator.setter
    def numerator(self, value):
        """Sets a new numerator and generates the ETA. Must be greater than or equal to previous numerator."""
        # If ETA is every iteration, don't do anything fancy.
        if self.eta_every <= 1:
            self._eta.numerator = value
            self._eta_string = self._generate_eta(self._eta.eta_seconds)
            return

        # If ETA is not every iteration, unstable rate is used. If this bar is
        # undefined, no point in calculating ever.
        if self._eta.undefined:
            self._eta.set_numerator(value, calculate=False)
            return

        # Calculate if this iteration is the right one.
        if self._eta_count >= self.eta_every:
            self._eta_count = 1
            self._eta.numerator = value
            self._eta_string = self._generate_eta(self._eta.eta_seconds)
            return

        self._eta_count += 1
        self._eta.set_numerator(value, calculate=False)

    @property
    def percent(self):
        """Returns the percent as a float."""
        return float(self._eta.percent)

    @property
    def rate(self):
        """Returns the rate of the progress as a float. Selects the unstable rate if eta_every > 1 for performance."""
        return float(self._eta.rate_unstable if self.eta_every > 1 else self._eta.rate)

    @property
    def undefined(self):
        """Return True if the progress bar is undefined (unknown denominator)."""
        return self._eta.undefined


class ProgressBar(BaseProgressBar):

    """Draw a progress bar showing the ETA, percentage, done/total items, and a spinner.

    Looks like one of these:
      8% (  8/100) [##                                  ] eta 00:24 /
    100% (100/100) [####################################] eta 00:01 -
    23 [                       ?                        ] eta --:-- |

    Positional arguments:
    denominator -- the final/total number of units (like the expected file size of a download). 0 if unknown.

    Keyword arguments:
    max_with -- limit number of characters shown (by default the full progress bar takes up the entire terminal width).

    Instance variables:
    template -- string template of the full progress bar.
    bar -- class instance of the 'bar' part of the full progress bar.

    More instance variables in etaprogress.components.base_progress_bar.BaseProgressBar.
    """

    def __init__(self, denominator, max_width=None, description=''):
        super(ProgressBar, self).__init__(denominator, max_width=max_width)
        self.description = description
        if self.undefined:
            self.template = '{numerator} {bar} eta --:-- {spinner}'
            self.bar = BarUndefinedAnimated()
        else:
            self.template = '{percent:3d}% ({fraction}) {bar} eta {eta} {spinner}'
            self.bar = Bar()

        if description:
            self.template = '{desc:.60}: ' + self.template

    def __str__(self):
        """Returns the fully-built progress bar and other data."""
        # Partially build out template.
        bar = '{bar}'
        spinner = next(SPINNER)
        if self.undefined:
            numerator = self.str_numerator
            template = self.template.format(desc=self.description,
                                            numerator=numerator,
                                            bar=bar, spinner=spinner)
        else:
            percent = int(self.percent)
            fraction = self.str_fraction
            eta = self._eta_string or '--:--'
            template = self.template.format(desc=self.description,
                                            percent=percent, fraction=fraction,
                                            bar=bar, eta=eta, spinner=spinner)

        # Determine bar width and finish.
        width = get_remaining_width(
            template.format(bar=''), self.max_width or None)
        bar = self.bar.bar(width, percent=self.percent)
        return template.format(bar=bar)

    @staticmethod
    def _generate_eta(seconds):
        """Returns a human readable ETA string."""
        return '' if seconds is None else eta_hms(seconds, always_show_minutes=True)

    @property
    def str_fraction(self):
        """Returns the fraction with additional whitespace."""
        if self.undefined:
            return None
        denominator = locale.format('%d', self.denominator, grouping=True)
        numerator = self.str_numerator.rjust(len(denominator))
        return '{0}/{1}'.format(numerator, denominator)

    @property
    def str_numerator(self):
        """Returns the numerator as a formatted string."""
        return locale.format('%d', self.numerator, grouping=True)


# bars
class BarUndefinedEmpty(object):

    """Simplest progress bar. Just a static empty bar."""

    CHAR_LEFT_BORDER = '['
    CHAR_RIGHT_BORDER = ']'
    CHAR_EMPTY = ' '

    def __init__(self):
        self._width_offset = len(
            self.CHAR_LEFT_BORDER) + len(self.CHAR_RIGHT_BORDER)

    def bar(self, width, **_):
        """Returns the completed progress bar.

        Positional arguments:
        width -- the width of the entire bar (including borders).
        """
        return self.CHAR_LEFT_BORDER + self.CHAR_EMPTY * (width - self._width_offset) + self.CHAR_RIGHT_BORDER


class Bar(BarUndefinedEmpty):

    """A regular progress bar."""

    CHAR_FULL = '#'
    CHAR_LEADING = '#'

    def bar(self, width, percent=0):
        """Returns the completed progress bar.

        Positional arguments:
        width -- the width of the entire bar (including borders).

        Keyword arguments:
        percent -- the percentage to draw.
        """
        width -= self._width_offset
        units = int(percent * 0.01 * width)
        if not units:
            return self.CHAR_LEFT_BORDER + self.CHAR_EMPTY * width + self.CHAR_RIGHT_BORDER

        final_bar = (
            self.CHAR_LEFT_BORDER +
            self.CHAR_FULL * (units - 1) +
            self.CHAR_LEADING +
            self.CHAR_EMPTY * (width - units) +
            self.CHAR_RIGHT_BORDER
        )
        return final_bar


class BarUndefinedAnimated(BarUndefinedEmpty):

    """Progress bar with a character that moves back and forth."""

    CHAR_ANIMATED = '?'

    def __init__(self):
        super(BarUndefinedAnimated, self).__init__()
        self._width_offset += len(self.CHAR_ANIMATED)
        self._position = -1
        self._direction = 1

    def bar(self, width, **_):
        """Returns the completed progress bar. Every time this is called the animation moves.

        Positional arguments:
        width -- the width of the entire bar (including borders).
        """
        width -= self._width_offset
        self._position += self._direction

        # Change direction.
        if self._position <= 0 and self._direction < 0:
            self._position = 0
            self._direction = 1
        elif self._position > width:
            self._position = width - 1
            self._direction = -1

        final_bar = (
            self.CHAR_LEFT_BORDER +
            self.CHAR_EMPTY * self._position +
            self.CHAR_ANIMATED +
            self.CHAR_EMPTY * (width - self._position) +
            self.CHAR_RIGHT_BORDER
        )
        return final_bar

# util functions


def eta_hms(seconds, always_show_hours=False, always_show_minutes=False, hours_leading_zero=False):
    """Converts seconds remaining into a human readable timestamp (e.g. hh:mm:ss, h:mm:ss, mm:ss, or ss).

    Positional arguments:
    seconds -- integer/float indicating seconds remaining.

    Keyword arguments:
    always_show_hours -- don't hide the 0 hours.
    always_show_minutes -- don't hide the 0 minutes.
    hours_leading_zero -- show 01:00:00 instead of 1:00:00.

    Returns:
    Human readable string.
    """
    # Convert seconds to other units.
    final_hours, final_minutes, final_seconds = 0, 0, seconds
    if final_seconds >= 3600:
        final_hours = int(final_seconds / 3600.0)
        final_seconds -= final_hours * 3600
    if final_seconds >= 60:
        final_minutes = int(final_seconds / 60.0)
        final_seconds -= final_minutes * 60
    final_seconds = int(ceil(final_seconds))

    # Determine which string template to use.
    if final_hours or always_show_hours:
        if hours_leading_zero:
            template = '{hour:02.0f}:{minute:02.0f}:{second:02.0f}'
        else:
            template = '{hour}:{minute:02.0f}:{second:02.0f}'
    elif final_minutes or always_show_minutes:
        template = '{minute:02.0f}:{second:02.0f}'
    else:
        template = '{second:02.0f}'
    return template.format(hour=final_hours, minute=final_minutes, second=final_seconds)


def get_remaining_width(sample_string, max_terminal_width=None):
    """Returns the number of characters available if sample string were to be printed in the terminal.

    Positional arguments:
    sample_string -- gets the length of this string.

    Keyword arguments:
    max_terminal_width -- limit the overall width of everything to these many characters.

    Returns:
    Integer.
    """
    if max_terminal_width is not None:
        available_width = min(terminal_width(), max_terminal_width)
    else:
        available_width = terminal_width()
    return available_width - len(sample_string)