import numpy as np

def _is_stochastic_matrix_impl(T, tol):
    dim = T.shape[0]
    X = np.abs(T) - T
    x = np.sum(T, axis = 1)
    return np.abs(x - np.ones(dim)).max() < dim * tol and X.max() < 2.0 * tol
