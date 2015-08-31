
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
Created on 21.04.2015

@author: marscher
'''

from __future__ import absolute_import

"""Miscellaneous classes/functions/etc."""
import os
import struct
import ctypes

if os.name != 'nt':
    import fcntl
    import termios
else:
    import ctypes.wintypes

DEFAULT_TERMINAL_WIDTH = None


class _WindowsCSBI(object):
    """Interfaces with Windows CONSOLE_SCREEN_BUFFER_INFO API/DLL calls. Gets info for stderr and stdout.

    References:
        https://code.google.com/p/colorama/issues/detail?id=47.
        pytest's py project: py/_io/terminalwriter.py.

    Class variables:
    CSBI -- ConsoleScreenBufferInfo class/struct (not instance, the class definition itself) defined in _define_csbi().
    HANDLE_STDERR -- GetStdHandle() return integer for stderr.
    HANDLE_STDOUT -- GetStdHandle() return integer for stdout.
    WINDLL -- my own loaded instance of ctypes.WinDLL.
    """

    CSBI = None
    HANDLE_STDERR = None
    HANDLE_STDOUT = None
    WINDLL = ctypes.LibraryLoader(getattr(ctypes, 'WinDLL', None))

    @staticmethod
    def _define_csbi():
        """Defines structs and populates _WindowsCSBI.CSBI."""
        if _WindowsCSBI.CSBI is not None:
            return

        class COORD(ctypes.Structure):
            """Windows COORD structure. http://msdn.microsoft.com/en-us/library/windows/desktop/ms682119"""
            _fields_ = [('X', ctypes.c_short), ('Y', ctypes.c_short)]

        class SmallRECT(ctypes.Structure):
            """Windows SMALL_RECT structure. http://msdn.microsoft.com/en-us/library/windows/desktop/ms686311"""
            _fields_ = [('Left', ctypes.c_short), ('Top', ctypes.c_short), ('Right', ctypes.c_short),
                        ('Bottom', ctypes.c_short)]

        class ConsoleScreenBufferInfo(ctypes.Structure):
            """Windows CONSOLE_SCREEN_BUFFER_INFO structure.
            http://msdn.microsoft.com/en-us/library/windows/desktop/ms682093
            """
            _fields_ = [
                ('dwSize', COORD),
                ('dwCursorPosition', COORD),
                ('wAttributes', ctypes.wintypes.WORD),
                ('srWindow', SmallRECT),
                ('dwMaximumWindowSize', COORD)
            ]

        _WindowsCSBI.CSBI = ConsoleScreenBufferInfo

    @staticmethod
    def initialize():
        """Initializes the WINDLL resource and populated the CSBI class variable."""
        _WindowsCSBI._define_csbi()
        _WindowsCSBI.HANDLE_STDERR = _WindowsCSBI.HANDLE_STDERR or _WindowsCSBI.WINDLL.kernel32.GetStdHandle(-12)
        _WindowsCSBI.HANDLE_STDOUT = _WindowsCSBI.HANDLE_STDOUT or _WindowsCSBI.WINDLL.kernel32.GetStdHandle(-11)
        if _WindowsCSBI.WINDLL.kernel32.GetConsoleScreenBufferInfo.argtypes:
            return

        _WindowsCSBI.WINDLL.kernel32.GetStdHandle.argtypes = [ctypes.wintypes.DWORD]
        _WindowsCSBI.WINDLL.kernel32.GetStdHandle.restype = ctypes.wintypes.HANDLE
        _WindowsCSBI.WINDLL.kernel32.GetConsoleScreenBufferInfo.restype = ctypes.wintypes.BOOL
        _WindowsCSBI.WINDLL.kernel32.GetConsoleScreenBufferInfo.argtypes = [
            ctypes.wintypes.HANDLE, ctypes.POINTER(_WindowsCSBI.CSBI)
        ]

    @staticmethod
    def get_info(handle):
        """Get information about this current console window (for Microsoft Windows only).

        Raises IOError if attempt to get information fails (if there is no console window).

        Don't forget to call _WindowsCSBI.initialize() once in your application before calling this method.

        Positional arguments:
        handle -- either _WindowsCSBI.HANDLE_STDERR or _WindowsCSBI.HANDLE_STDOUT.

        Returns:
        Dictionary with different integer values. Keys are:
            buffer_width -- width of the buffer (Screen Buffer Size in cmd.exe layout tab).
            buffer_height -- height of the buffer (Screen Buffer Size in cmd.exe layout tab).
            terminal_width -- width of the terminal window.
            terminal_height -- height of the terminal window.
            bg_color -- current background color (http://msdn.microsoft.com/en-us/library/windows/desktop/ms682088).
            fg_color -- current text color code.
        """
        # Query Win32 API.
        csbi = _WindowsCSBI.CSBI()
        try:
            if not _WindowsCSBI.WINDLL.kernel32.GetConsoleScreenBufferInfo(handle, ctypes.byref(csbi)):
                raise IOError('Unable to get console screen buffer info from win32 API.')
        except ctypes.ArgumentError:
            raise IOError('Unable to get console screen buffer info from win32 API.')

        # Parse data.
        result = dict(
            buffer_width=int(csbi.dwSize.X - 1),
            buffer_height=int(csbi.dwSize.Y),
            terminal_width=int(csbi.srWindow.Right - csbi.srWindow.Left),
            terminal_height=int(csbi.srWindow.Bottom - csbi.srWindow.Top),
            bg_color=int(csbi.wAttributes & 240),
            fg_color=int(csbi.wAttributes % 16),
        )
        return result


def terminal_width():
    """Returns the terminal's width (number of character columns)."""
    try:
        if os.name == 'nt':
            _WindowsCSBI.initialize()
            return _WindowsCSBI.get_info(_WindowsCSBI.HANDLE_STDOUT)['terminal_width']
        return struct.unpack('hhhh', fcntl.ioctl(0, termios.TIOCGWINSZ, '\000' * 8))[1]
    except IOError:
        return 80