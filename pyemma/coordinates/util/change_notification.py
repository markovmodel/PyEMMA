
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
Created on 14.01.2016

@author: marscher
'''


def inform_children_upon_change(f):
    """ decorator to call interface method '_stream_upon_change' of NotifyOnChangesMixin
    instances """

    def _notify(self, *args, **kw):
        # first call the decorated function, then inform about the change.
        res = f(self, *args, **kw)
        self._stream_on_change()
        return res

    return _notify


class NotifyOnChangesMixIn(object):
    #### interface to handle events

    @property
    def _stream_children(self):
        if not hasattr(self, "_stream_children_list"):
            self._stream_children_list = []
        return self._stream_children_list

    def _stream_register_child(self, data_producer):
        """ should be called upon setting of data_producer """
        self._stream_children.append(data_producer)

    def _stream_unregister_child(self, child):
        try:
            self._stream_children.remove(child)
        except ValueError:
            print("should not happen")

    def _stream_on_change(self):
        pass

    def _stream_inform_children_param_change(self):
        """ will inform all children about a parameter change in general """
        for c in self._stream_children:
            c._stream_on_change()
