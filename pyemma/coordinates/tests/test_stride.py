import unittest
import os
import tempfile
import numpy as np
import mdtraj
import pyemma


class TestStride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = super(TestStride, cls).setUpClass()

        cls.dim = 99  # dimension (must be divisible by 3) # 99
        N_trajs = 10  # number of trajectories

        # create topology file
        cls.temppdb = tempfile.mktemp('.pdb')
        with open(cls.temppdb, 'w') as f:
            for i in xrange(cls.dim//3):
                print>>f, ('ATOM  %5d C    ACE A   1      28.490  31.600  33.379  0.00  1.00' % i)
        
        cls.trajnames = []  # list of xtc file names
        cls.data = []
        for i in xrange(N_trajs):
            # set up data
            N = int(np.random.rand()*1000+10)
            xyz = np.random.randn(N, cls.dim//3, 3).astype(np.float32)
            cls.data.append(xyz)
            t = np.arange(0, N)
            # create trajectory file
            traj = mdtraj.load(cls.temppdb)
            traj.xyz = xyz
            traj.time = t
            tempfname = tempfile.mktemp('.xtc')
            traj.save(tempfname)
            cls.trajnames.append(tempfname)

    def test_length_and_content_feature_reader_and_TICA(self):
        for stride in xrange(1, 100, 20):
            r = pyemma.coordinates.feature_reader(self.trajnames, self.temppdb)
            t = pyemma.coordinates.transform.tica.TICA(2, 2, force_eigenvalues_le_one=True)
            t.data_producer = r
            t.parametrize()
            
            # subsample data
            out_tica = t.get_output(stride=stride)
            out_reader = r.get_output(stride=stride)
            
            # get length in different ways
            len_tica = [x.shape[0] for x in out_tica]
            len_reader = [x.shape[0] for x in out_reader]
            len_trajs = t.trajectory_lengths(stride=stride)
            len_ref = [(x.shape[0]-1)//stride+1 for x in self.data]
            # print 'len_ref', len_ref
            
            # compare length
            self.assertTrue(len_ref == len_trajs)
            self.assertTrue(len_ref == len_tica)
            self.assertTrue(len_ref == len_reader)
            
            # compare content (reader)
            for ref_data, test_data in zip(self.data, out_reader):
                ref_data_reshaped = ref_data.reshape((ref_data.shape[0], ref_data.shape[1]*3))
                self.assertTrue(np.allclose(ref_data_reshaped[::stride, :], test_data, atol=1E-3))

    def test_content_data_in_memory(self):
        # prepare test data
        N_trajs = 10
        d = []
        for _ in xrange(N_trajs):
            N = int(np.random.rand()*1000+10)
            d.append(np.random.randn(N, 10).astype(np.float32))
        
        # read data
        reader = pyemma.coordinates.memory_reader(d)
        
        # compare
        for stride in xrange(1, 10):
            out_reader = reader.get_output(stride=stride)
            for ref_data, test_data in zip(d, out_reader):
                self.assertTrue(np.all(ref_data[::stride] == test_data))  # here we can test exact equality
                
    def test_parametrize_with_stride(self):
        # for stride in xrange(1,100,20):
        for stride in xrange(1, 100, 5):
            r = pyemma.coordinates.feature_reader(self.trajnames, self.temppdb)
            # print 'expected total length of trajectories:', r.trajectory_lengths(stride=stride)
            tau = 5
            # print 'expected inner frames', [max(l-2*tau,0) for l in r.trajectory_lengths(stride=stride)]
            t = pyemma.coordinates.transform.tica.TICA(tau=tau, output_dimension=2, force_eigenvalues_le_one=True)
            # force_eigenvalues_le_one=True enables an internal consitency check in TICA
            t.data_producer = r
            # print 'STRIDE:', stride
            # print 'theoretical result 2*(N-tau):', sum([2*(x-5) for x in r.trajectory_lengths(stride=stride) if x > 5])
            # print 'theoretical result N:', sum(r.trajectory_lengths(stride=stride))
            t.parametrize(stride=stride)
            # print 'TICA', t.N_cov, 2*t.N_cov_tau
            # print 'eigenvalues', sorted(t.eigenvalues)[::-1][0:5]
            self.assertTrue(np.all(t.eigenvalues <= 1.0+1.E-12))

    @classmethod
    def tearDownClass(cls):
        for fname in cls.trajnames:
            os.unlink(fname)
        os.unlink(cls.temppdb)
        super(TestStride, cls).tearDownClass()            

if __name__ == "__main__":
    unittest.main()
