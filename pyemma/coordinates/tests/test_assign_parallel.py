"""
This script is invoked by test_assign.py since open mp does only
support setting

"""

import pyemma.coordinates as coor
import os
import numpy as np

def worker():
    assert os.environ['OMP_NUM_THREADS'] == "4"
    X = np.random.random((10000, 3))
    centers = X[np.random.choice(10000, 10)]
    res = coor.assign_to_centers(X, centers, n_jobs=2, return_dtrajs=False)
    assert res.n_jobs == int(os.environ['OMP_NUM_THREADS']), res.n_jobs


if __name__ == '__main__':
    worker()