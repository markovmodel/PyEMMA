'''
Created on 22.01.2015

@author: marscher
'''
import os
import unittest

from coordinates.coordinate_transformation.discretizer import Discretizer
from coordinates.coordinate_transformation.feature_reader import FeatureReader
from coordinates.coordinate_transformation.featurizer import Featurizer
from coordinates.coordinate_transformation.tica import TICA
from coordinates.coordinate_transformation.writer import WriterCSV
import numpy as np


class TestWriterCSV(unittest.TestCase):

    def setUp(self):
        trajfiles = ['/home/marscher/kchan/traj01_sliced.xtc']
        topfile = '/home/marscher/kchan/Traj_Structure.pdb'

        # create featurizer
        self.featurizer = Featurizer(topfile)
        sel = np.array([(0, 20), (200, 320), (1300, 1500)])
        self.featurizer.distances(sel)
        # feature reader
        self.reader = FeatureReader(trajfiles, topfile, self.featurizer)

        self.output_file = 'test_writer_csv.dat'

    def tearDown(self):
        # print "delete output"
        os.unlink(self.output_file)

    def testWriter(self):
        self.writer = WriterCSV(self.output_file, self.reader)
        self.D = Discretizer([self.reader, self.writer])

        self.reader.operate_in_memory()
        self.D.run()

        # open file and compare to reader.Y
        output = np.loadtxt(self.output_file)
        np.testing.assert_allclose(output, self.reader.Y[0])

    def testTicaWriter(self):
        self.writer = WriterCSV(self.output_file, self.reader)
        tica = TICA(self.reader, 10, 2)
        self.D = Discretizer([self.reader, tica, self.writer])

        self.reader.operate_in_memory()
        self.D.run()
        
        # open file and compare to reader.Y
        output = np.loadtxt(self.output_file)
        np.testing.assert_allclose(output, self.reader.Y[0])

if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
