# This file is part of thermotools.
#
# Copyright 2015 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# thermotools is free software: you can redistribute it and/or modify
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

r"""
This module provides a custom exception to interrupt estimation runs via callback functions.
"""

__all__ = [
    'CallbackInterrupt',
    'generic_callback_stop']

class CallbackInterrupt(Exception):
    r"""
    Exception class estimation interruptions via callback functions.
    """
    def __init__(self, msg):
        self.msg = msg
    def __str__(self):
        return "[CALLBACKINTERRUPT] %s" % self.msg

def generic_callback_stop(**kwargs):
    r"""
    This is a generic callback serving as example and for testing purposes: it just stops
    the estimation at first evaluation of the callback function.
    """
    raise CallbackInterrupt("STOP")
