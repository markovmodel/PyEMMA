'''
Created on 22.01.2014

@author: marscher
'''
import unittest

import emma2.util.pystallone as st
import numpy as np

class TestPyStallone(unittest.TestCase):
    def setUp(self):
        # setup a random numpy array for each test case
        self.n = 1000
        self.a = np.random.random(self.n)

    def convertToNPandCompare(self, stArray, npArray, skipDataTypeCheck=False):
        """
        converts stArray to a numpy array and compares with npArray
        """
        newNPArr = st.stallone_array_to_ndarray(stArray)
        self.assertEqual(type(npArray), type(newNPArr), 'differing types')
        self.assertEqual(npArray.shape, newNPArr.shape, 'differing shape')
        self.assertTrue(np.allclose(npArray, newNPArr), 'conversion failed')
        if not skipDataTypeCheck:
            self.assertEqual(npArray.dtype, newNPArr.dtype, 
                         'differing datatypes: is %s, should be %s' 
                          %(npArray.dtype, newNPArr.dtype))

        
    def compareNP(self, a, b):
        self.assertEqual(type(a), type(b), 'differing types: should be %s, but is %s' 
                         % (type(a), type(b)))
        self.assertEqual(a.dtype, b.dtype, 'differing datatypes:'\
                         ' should be %s, but is %s' % (a.dtype, b.dtype))
        self.assertEqual(a.shape, b.shape, 'differing shape: should be %s, but is %s'
                         % (a.shape, b.shape))
        self.assertTrue(np.allclose(a, b), 'conversion failed')


    def testConversionND_Float32(self):
        a = self.a.astype(np.float32)
        b = st.ndarray_to_stallone_array(a)
        self.convertToNPandCompare(b, a, True)

    def testConversionND_Float64(self):
        a = self.a.astype(np.float64)
        b = st.ndarray_to_stallone_array(a)
        self.convertToNPandCompare(b, a)


    def testConversionND_Int32(self):
        a = self.a.astype(np.int32)
        b = st.ndarray_to_stallone_array(a)
        self.convertToNPandCompare(b, a, True)
        
    @unittest.expectedFailure # since stallone only maps int32, input converted.
    def testConversionND_Int64(self):
        a = self.a.astype(np.int64)
        b = st.ndarray_to_stallone_array(a)
        self.convertToNPandCompare(b, a)

    def testIDoubleArray2ND_double1d(self):
        jarr = st.JArray(st.JDouble)(self.a)
        a = st.API.doublesNew.array(jarr)
        b = st.stallone_array_to_ndarray(a)
        self.compareNP(self.a, b)
    
    def testIDoubleArray2ND_double2d(self):
        self.a = self.a.reshape(( 10, self.n/ 10 ))
        # create 2d Jarray
        jarr = st.JArray(st.JDouble, 2)(self.a)
        # wrap it
        a = st.API.doublesNew.array(jarr)
        # convert back
        b = st.stallone_array_to_ndarray(a)
        
        self.compareNP(self.a, b)

    def testJArr2ND_int1d(self):
        self.a = np.random.randint(1, size=self.n).astype(np.int32)
        jarr = st.JArray(st.JInt)(self.a)
         
        a = st.API.intsNew.arrayFrom(jarr)
        b = st.stallone_array_to_ndarray(a)

        self.compareNP(self.a, b)
    
    def testJArr2ND_int2d(self):
        self.a = np.random.randint(1, size=self.n).reshape((10, self.n/10)).astype(np.int32)
        #self.a = self.a.reshape((10, self.n / 10))
        list_of_subarrays = [st.JArray(st.JInt, 1)(r) for r in self.a]
        jarr = st.JArray(st.JInt, 2)(list_of_subarrays)
        # directly invoke factory, which acceppts only int32
        a = st.API.intsNew.table(jarr)
        b = st.stallone_array_to_ndarray(a)
        self.compareNP(self.a, b)

    @unittest.expectedFailure # known to be broken...
    def testDirectBufferFloat(self):
        stArr = st.ndarray_to_stallone_array(self.a, copy=False)
        # make change in self.a and see if its reflected in stallone array
        self.a[0] = 42
        self.assertEqual(42, stArr.get(0))

if __name__ == "__main__":
    unittest.main()
