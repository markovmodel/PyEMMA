'''
Test feature reader and Tica by checking the properties of the ICs.
cov(ic_i,ic_j) = delta_ij and cov(ic_i,ic_j,tau) = lambda_i delta_ij
@author: Fabian Paul
'''
import unittest
import os
import tempfile
import numpy as np
import numpy.random
import mdtraj
from pyemma.coordinates.api import feature_reader, tica, _TICA as TICA
from pyemma.coordinates.io.feature_reader import FeatureReader
from pyemma.util.log import getLogger

log = getLogger('TestFeatureReaderAndTICA')

def save_trajs(trans):
    fnames = []
    last_itraj = -1
    f = None
    log.info('ic dimension: %d'%trans.dimension())
    for itraj,chunk in trans:
        #print 'chsh:',chunk.shape
        if itraj!=last_itraj:
            if f!=None:
                f.close()
            fname = tempfile.mktemp('.dat')
            fnames.append(fname)
            f = open(fname,'w')
            
        np.savetxt(f,chunk)
        last_itraj = itraj
        
    if f!=None:
        f.close()
    return fnames

def random_invertible(n,eps=0.01):
    'generate real random invertible matrix'
    m=np.random.randn(n,n)
    u,s,v=np.linalg.svd(m)
    s=np.maximum(s,eps)
    return u.dot(np.diag(s)).dot(v)

from nose.plugins.attrib import attr
@attr(slow=True)
class TestFeatureReaderAndTICA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        c = super(TestFeatureReaderAndTICA, cls).setUpClass()

        cls.dim = 99 # dimension (must be divisible by 3)
        N = 5000 # length of single trajectory # 500000
        N_trajs = 10 # number of trajectories

        A = random_invertible(cls.dim) # mixing matrix
        # tica will approximate its inverse with the projection matrix
        mean = np.random.randn(cls.dim)

        # create topology file
        cls.temppdb = tempfile.mktemp('.pdb')
        with open(cls.temppdb,'w') as f:
            for i in xrange(cls.dim//3):
                print>>f, \
                  ('ATOM  %5d C    ACE A   1      28.490  31.600  33.379  0.00  1.00'%i)

        t = np.arange(0,N)
        cls.trajnames = [] # list of xtc file names
        for i in xrange(N_trajs):
            # set up data
            white = np.random.randn(N,cls.dim)
            brown = np.cumsum(white,axis=0)
            correlated = np.dot(brown,A)
            data = correlated + mean
            xyz = data.reshape((N,cls.dim//3,3))
            # create trajectory file
            traj = mdtraj.load(cls.temppdb)
            traj.xyz = xyz
            traj.time = t
            tempfname = tempfile.mktemp('.xtc')
            traj.save(tempfname)
            cls.trajnames.append(tempfname)

    @classmethod
    def tearDownClass(cls):
        for fname in cls.trajnames:
            os.unlink(fname)
        os.unlink(cls.temppdb)
        super(TestFeatureReaderAndTICA, cls).tearDownClass()
        
    def test_covariances_and_eigenvalues(self):
        reader = FeatureReader(self.trajnames, self.temppdb)
        trans = TICA(tau=1,output_dimension=self.dim, force_eigenvalues_le_one=True)
        trans.data_producer = reader
        for tau in [1,10,100,1000,2000]:
            log.info('number of trajectories reported by tica %d'%trans.number_of_trajectories())
            trans.tau = tau
            ic_fnames = save_trajs(trans) # this runs again the chain after the change of tau
            #print '@@cov', trans.cov
            #print '@@cov_tau', trans.cov_tau

            log.info('max. eigenvalue: %f'%np.max(trans.eigenvalues))
            self.assertTrue(np.all(trans.eigenvalues <= 1.0))
            
            # check ICs
            check = tica(data=ic_fnames, lag=tau, dim=self.dim, force_eigenvalues_le_one=True)
            _ = iter(check) # grab iterator to run the transform
            self.assertTrue(np.allclose(np.eye(self.dim),check.cov))
            ic_cov_tau = np.zeros((self.dim,self.dim))
            ic_cov_tau[np.diag_indices(self.dim)] = trans.eigenvalues
            self.assertTrue(np.allclose(ic_cov_tau,check.cov_tau))
            #print '@@cov_tau', check.cov_tau
            
            for fname in ic_fnames:
                os.unlink(fname)

if __name__ == "__main__":
    unittest.main()

