'''
Created on Jun 20, 2014

@author: marscher
'''
import unittest
import os

# ipy notebook runner
from runipy.notebook_runner import NotebookRunner
from IPython.nbformat.current import read
from IPython.nbformat.v3.nbbase import new_code_cell

# point to ipython directory in pyemma
ipy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def case_generator(filename):
    
    # currently there is some bug in parallel execution of ipython, so we disable all tests by now.
    @unittest.SkipTest
    def test(self):
        payload = open(filename)
        nb = read(payload, 'json')
        # turn off colors, so we can properly read exceptions
        cell_nc = new_code_cell(input='%colors NoColor', prompt_number=0)
        nb.worksheets[0].cells.insert(0, cell_nc)
        
        # workaround for matplotlib backend
        cell_mpl = new_code_cell(input="import matplotlib; matplotlib.use('Agg')", prompt_number=1)
        nb.worksheets[0].cells.insert(1, cell_mpl)
        
        # set working dir to notebook path
        wd = os.path.abspath(os.path.dirname(filename))
        runner = NotebookRunner(nb, working_dir=wd)
        #try:
        runner.run_notebook(skip_exceptions=False)
        #except Exception as e:
        #    exc_type, exc_obj, exc_tb = sys.exc_info()
        #    self.fail('Notebook %s threw an exception of type %s:\n(%s)' %
        #             (filename, exc_type, e))
    return test

class TestNotebooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super(TestNotebooks, cls).setUpClass()
        for p, d, filenames in os.walk(ipy_dir):
            for f in filenames:
                if f.endswith('.ipynb') and f.rfind('-checkpoint') == -1:
                    test_name = 'test_%s' % os.path.splitext(f)[0]
                    test = case_generator(os.path.join(p, f))
                    setattr(cls, test_name, test)

    def testDummy(self):
        """
        this is needed for py.test, so the class is considered during test collection
        Do not remove!
        """
        pass

if __name__ == '__main__':
    unittest.main()
