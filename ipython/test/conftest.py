'''
Created on Jun 20, 2014

@author: marscher
'''

def pytest_pycollect_makeitem(collector, name, obj):
    """
    this is a hook for pytest to enforce the dynamic generated testcases
    of 'TestNotebooks' are initialized.
    """
    if name == 'TestNotebooks':
        obj.setUpClass()
