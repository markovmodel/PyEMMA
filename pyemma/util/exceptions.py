
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
Created on May 26, 2014

@author: marscher
'''


class SpectralWarning(RuntimeWarning):
    pass


class ImaginaryEigenValueWarning(SpectralWarning):
    pass


class PrecisionWarning(RuntimeWarning):
    r"""
    This warning indicates that some operation in your code leads
    to a conversion of datatypes, which involves a loss/gain in
    precision.

    """
    pass


class NotConvergedWarning(RuntimeWarning):
    r"""
    This warning indicates that some iterative procdure has not
    converged or reached the maximum number of iterations implemented
    as a safe guard to prevent arbitrary many iterations in loops with
    a conditional termination criterion.

    """
    pass


class EfficiencyWarning(UserWarning):
    r"""Some operation or input data leads to a lack of efficiency"""
    pass


class ParserWarning(UserWarning):
    """ Some user defined variable could not be parsed and is ignored/replaced. """
    pass


class ConfigDirectoryException(Exception):
    """ Some operation with PyEMMAs configuration directory went wrong. """
    pass


class PyEMMA_DeprecationWarning(UserWarning):
    """You are using a feature, which will be removed in a future release. You have been warned!"""
    pass