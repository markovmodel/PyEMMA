"""Easy to use ETA calculation. Used by etaprogress.progress.
taken from:

https://github.com/Robpol86/etaprogress
https://pypi.python.org/pypi/etaprogress
"""

from __future__ import division
from collections import deque
from math import sqrt
import time

__all__ = ('ETA', )
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

    def __str__(self):
        res = "ETA:\t"
        if self.eta_epoch is None:
            res += 'unknown.'
        else:
            res += str(int(max([self.eta_epoch - _NOW(), 0]))) + 's.'
        res += ' Currently at:\t' + str(int(self.percent)) + '%'
        return res
