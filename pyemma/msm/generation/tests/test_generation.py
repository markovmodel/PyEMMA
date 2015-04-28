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
@author: noe
'''
import unittest
import numpy as np
import pyemma.msm.generation as msmgen
import pyemma.msm.estimation as msmest
import pyemma.msm.analysis as msmana

class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_trajectory(self):
        P = np.array([[0.9,0.1],
                      [0.1,0.9]])
        N = 1000
        traj = msmgen.generate_traj(P, N, start=0)

        # test shapes and sizes
        assert traj.size == N
        assert traj.min() >= 0
        assert traj.max() <= 1

        # test statistics of transition matrix
        C = msmest.count_matrix(traj,1)
        Pest = msmest.transition_matrix(C)
        assert np.max(np.abs(Pest - P)) < 0.025


    def test_trajectories(self):
        P = np.array([[0.9,0.1],
                      [0.1,0.9]])

        # test number of trajectories
        M = 10
        N = 10
        trajs = msmgen.generate_trajs(P, M, N, start=0)
        assert len(trajs) == M

        # test statistics of starting state
        trajs = msmgen.generate_trajs(P, 1000, 1)
        ss = np.concatenate(trajs).astype(int)
        pi = msmana.stationary_distribution(P)
        piest = msmest.count_states(ss) / 1000.0
        assert np.max(np.abs(pi - piest)) < 0.025

        # test stopping state = starting state
        M = 10
        trajs = msmgen.generate_trajs(P, M, N, start=0, stop=0)
        for traj in trajs:
            assert traj.size == 1

        # test if we always stop at stopping state
        M = 100
        stop = 1
        trajs = msmgen.generate_trajs(P, M, N, start=0, stop=stop)
        for traj in trajs:
            assert traj.size == N or traj[-1] == stop
            assert stop not in traj[:-1]

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
