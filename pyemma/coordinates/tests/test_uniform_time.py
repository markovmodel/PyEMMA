
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Free University
# Berlin, 14195 Berlin, Germany.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
#  * Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation and/or
# other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

'''
Created on 09.04.2015

@author: marscher
'''
import unittest

import numpy as np

from pyemma.coordinates import api
from pyemma.coordinates.data.data_in_memory import DataInMemory


class TestUniformTimeClustering(unittest.TestCase):

    def test_1d(self):
        x = np.random.random(1000)
        reader = DataInMemory(x)

        k = 2
        c = api.cluster_uniform_time(k=k)

        c.data_producer = reader
        c.parametrize()

    def test_2d(self):
        x = np.random.random((300, 3))
        reader = DataInMemory(x)

        k = 2
        c = api.cluster_uniform_time(k=k)

        c.data_producer = reader
        c.parametrize()

    def test_big_k(self):
        # TODO: fix this (some error handling should be done in _param_init)
        x = np.random.random((300, 3))
        reader = DataInMemory(x)

        k = 298
        c = api.cluster_uniform_time(k=k)

        c.data_producer = reader
        c.parametrize()


if __name__ == "__main__":
    unittest.main()