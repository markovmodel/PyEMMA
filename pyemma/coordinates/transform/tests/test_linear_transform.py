'''
Created on Jun 23, 2014

@author: marscher
'''
import unittest
import numpy as np
from pyemma.util.pystallone import API, ndarray_to_stallone_array,\
    stallone_array_to_ndarray

class TestTransformation(unittest.TestCase):

    def testLinearOperatorTransformation(self):
        n = 669
        # note transpose
        A = np.arange(2*n).reshape((n, 2)).astype(np.float).T
        b = np.arange(n).astype(np.float)

        expected = np.dot(A, b)
        
        A_st=ndarray_to_stallone_array(A)
        assert A_st.rows() == A.shape[0]
        assert A_st.columns() == A.shape[1]
        
        b=b.reshape((n/3, 3))
        b_st = ndarray_to_stallone_array(b)
        assert b_st.rows() == b.shape[0]
        
        # currently there is no python api wrapping this, so call it directly.
        transformation = API.coorNew.linear_operator(A_st)
        result = transformation.transform(b_st)
        
        # compare result
        result_nd = stallone_array_to_ndarray(result)
        assert np.allclose(expected, result_nd)
        
if __name__ == "__main__":
    unittest.main()