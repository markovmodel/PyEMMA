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
Test deprecated function regroup_DISK.
@author: Fabian Paul
'''
import unittest
import tempfile
import os
import numpy as np
import mdtraj
import pyemma.msm.util.mapping


class TestMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = super(TestMapping, cls).setUpClass()
        
        cls.N_clusters = 200
        cls.cluster_repeat = 25
        cls.N_frames = cls.cluster_repeat*cls.N_clusters  # length of single trajectory 
        cls.N_trajs = 10  # number of trajectories

        # create topology file
        cls.pdb_fname = tempfile.mktemp('.pdb')
        with open(cls.pdb_fname, 'w') as f:
            print>>f, ('ATOM  00001 C    ACE A   1      28.490  31.600  33.379  0.00  1.00')
            
        # create disctrajs
        cls.disctrajs = []
        for i in xrange(cls.N_trajs):
            #disctraj = np.random.randint(0,high=cls.N_clusters,size=cls.N_frames)
            disctraj = np.arange(0,cls.N_clusters,dtype=int).repeat(cls.cluster_repeat)
            np.random.shuffle(disctraj)
            cls.disctrajs.append(disctraj)
            
        # create input trajectories
        cls.xtc_fnames = []
        for i in xrange(cls.N_trajs):
            xyz = np.zeros((cls.N_frames,1,3),dtype=np.int32)
            xyz[:,0,:] = cls.disctrajs[i][:,np.newaxis]
            xtc = mdtraj.load(cls.pdb_fname)
            xtc_fname = tempfile.mktemp('.xtc')
            xtc.xyz = xyz
            xtc.time = np.arange(0, cls.N_frames)
            xtc.save(xtc_fname)
            cls.xtc_fnames.append(xtc_fname)
            
        # create directory for output files
        cls.outdir = tempfile.mkdtemp()

    def test_regroup_DISK(self):
        # run function
        cl_fnames = pyemma.msm.util.mapping.regroup_DISK(self.xtc_fnames, self.pdb_fname, self.disctrajs, self.outdir)
        
        # check xtcs
        for i in xrange(self.N_clusters):
            fname = self.outdir+os.sep+'%d.xtc'%i
            self.assertTrue(fname in cl_fnames)
            data = mdtraj.load(fname, top=self.pdb_fname)
            self.assertTrue(np.allclose(data.xyz, i))
            self.assertTrue(data.xyz.shape[0]==self.cluster_repeat*self.N_trajs)

    @classmethod
    def tearDownClass(cls):
        for f in cls.xtc_fnames:
            os.unlink(f)
        for i in xrange(cls.N_clusters):
            os.unlink(cls.outdir+os.sep+'%d.xtc'%i)
        os.rmdir(cls.outdir)

if __name__ == "__main__":
    unittest.main()
