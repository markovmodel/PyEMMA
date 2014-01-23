'''
Created on 22.01.2014

@author: marscher
'''
import unittest

import pystallone as st
import numpy as np

class TestPyStallone(unittest.TestCase):
    def setUp(self):
        # setup a random numpy array for each test case
        self.n = 100
        self.a = np.random.random(self.n)
        
    def testConversionND_Float32(self):
        a = self.a.astype(np.float32)
        b = st.ndarray_to_stallone_array(a)

    def testConversionND_Float64(self):
        a = self.a.astype(np.float64)
        b = st.ndarray_to_stallone_array(a)
        
    def testConversionND_Int32(self):
        a = self.a.astype(np.int32)
        b = st.ndarray_to_stallone_array(a)
    
    def testConversionND_Int64(self):
        a = self.a.astype(np.int64)
        b = st.ndarray_to_stallone_array(a)
        
    def testIDoubleArray2ND_double1d(self):
        jarr = st.JArray(st.JDouble)(self.a[:])
        a = st.API.doublesNew.array(jarr)
        b = st.stallone_array_to_ndarray(a)
        self.assertEqual(type(self.a), type(b), 'differing types')
        self.assertEqual(self.a.dtype, b.dtype, 'differing datatypes')
        self.assertEqual(self.a.shape, b.shape, 'differing shape')
        self.assertTrue(np.allclose(self.a, b), 'conversion failed')
        
    def testIDoubleArray2ND_double2d(self):
        self.a = self.a.reshape(( 10, self.n/ 10 ))
        # create 2d Jarray
        #rows = [st.JArray(st.JDouble)(r) for r in self.a]
        #jarr = st.JArray(st.JObject)(rows)
        jarr = st.JArray(st.JDouble, 2)(self.a)
        # wrap it
        a = st.API.doublesNew.array(jarr)
        # convert back
        b = st.stallone_array_to_ndarray(a)
        self.assertEqual(type(self.a), type(b), 'differing types')
        self.assertEqual(self.a.dtype, b.dtype, 'differing datatypes')
        self.assertEqual(self.a.shape, b.shape, 'differing shape')
        self.assertTrue(np.allclose(self.a, b), 'conversion failed')
        
    def testJArr2ND_int1d(self):
        self.a = np.random.randint(1, size=self.n)
        jarr = st.JArray(st.JInt)(self.a[:])
        
        a = st.API.intsNew.arrayFrom(jarr)
        b = st.stallone_array_to_ndarray(a)
        
        self.assertEqual(type(self.a), type(b), 'differing types')
        self.assertEqual(self.a.dtype, b.dtype, 'differing datatypes:'\
                         ' should be %s, but is %s' % (self.a.dtype, b.dtype))
        self.assertEqual(self.a.shape, b.shape, 'differing shape')
        self.assertTrue(np.allclose(self.a, b), 'conversion failed')
        
    def testJArr2ND_int2d(self):
        self.a = np.random.randint(1, size=self.n)
        self.a = self.a.reshape((10, self.n / 10))
        jarr = st.JArray(st.JInt, 2)(self.a)
        # FIXME: 2d jarray not compatible with table(int[][]) prototype?!
        a = st.API.intsNew.table(jarr)
        b = st.stallone_array_to_ndarray(a)
        self.assertEqual(type(self.a), type(b), 'differing types')
        self.assertEqual(self.a.dtype, b.dtype, 'differing datatypes')
        self.assertEqual(self.a.shape, b.shape, 'differing shape')
        self.assertTrue(np.allclose(self.a, b), 'conversion failed')

if __name__ == "__main__":
    unittest.main()