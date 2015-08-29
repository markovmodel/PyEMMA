################################################################################
#
#   dtram.pyx - dTRAM implementation in C (cython wrapper)
#
#   author: Christoph Wehmeyer <christoph.wehmeyer@fu-berlin.de>
#
################################################################################

import numpy as np
cimport numpy as np

cdef extern from "_dtram.h":
    void _log_nu_K_i_setter(
            double *log_nu_K_i,
            int *C_K_ij,
            int n_therm_states,
            int n_markov_states
        )
    void _log_nu_K_i_equation(
            double *log_nu_K_i,
            double *b_K_i,
            double *f_i,
            int *C_K_ij,
            int n_therm_states,
            int n_markov_states,
            double *scratch_j,
            double *new_log_nu_K_i
        )
    void _f_i_equation(
            double *log_nu_K_i,
            double *b_K_i,
            double *f_i,
            int *C_K_ij,
            int n_therm_states,
            int n_markov_states,
            double *scratch_K_j,
            double *scratch_j,
            double *new_f_i
        )
    void _p_K_ij_equation(
            double *log_nu_K_i,
            double *b_K_i,
            double *f_i,
            int *C_K_ij,
            int n_therm_states,
            int n_markov_states,
            double *scratch_j,
            double *p_K_ij
        )

    void _f_K_equation(
            double *b_K_i,
            double *f_i,
            int n_therm_states,
            int n_markov_states,
            double *scratch_j,
            double *f_K
        )

def log_nu_K_i_setter(
        np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
        np.ndarray[int, ndim=3, mode="c"] C_K_ij not None
    ):
    _log_nu_K_i_setter(
            <double*> np.PyArray_DATA( log_nu_K_i ),
            <int*> np.PyArray_DATA( C_K_ij ),
            log_nu_K_i.shape[0],
            log_nu_K_i.shape[1]
        )

def log_nu_K_i_equation(
        np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
        np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
        np.ndarray[double, ndim=1, mode="c"] f_i not None,
        np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
        np.ndarray[double, ndim=1, mode="c"] scratch_j not None,
        np.ndarray[double, ndim=2, mode="c"] new_log_nu_K_i not None
    ):
    _log_nu_K_i_equation(
            <double*> np.PyArray_DATA( log_nu_K_i ),
            <double*> np.PyArray_DATA( b_K_i ),
            <double*> np.PyArray_DATA( f_i ),
            <int*> np.PyArray_DATA( C_K_ij ),
            log_nu_K_i.shape[0],
            log_nu_K_i.shape[1],
            <double*> np.PyArray_DATA( scratch_j ),
            <double*> np.PyArray_DATA( new_log_nu_K_i )
        )

def f_i_equation(
        np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
        np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
        np.ndarray[double, ndim=1, mode="c"] f_i not None,
        np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
        np.ndarray[double, ndim=2, mode="c"] scratch_K_j not None,
        np.ndarray[double, ndim=1, mode="c"] scratch_j not None,
        np.ndarray[double, ndim=1, mode="c"] new_f_i not None
    ):
    _f_i_equation(
            <double*> np.PyArray_DATA( log_nu_K_i ),
            <double*> np.PyArray_DATA( b_K_i ),
            <double*> np.PyArray_DATA( f_i ),
            <int*> np.PyArray_DATA( C_K_ij ),
            log_nu_K_i.shape[0],
            log_nu_K_i.shape[1],
            <double*> np.PyArray_DATA( scratch_K_j ),
            <double*> np.PyArray_DATA( scratch_j ),
            <double*> np.PyArray_DATA( new_f_i )
        )

def p_K_ij_equation(
        np.ndarray[double, ndim=2, mode="c"] log_nu_K_i not None,
        np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
        np.ndarray[double, ndim=1, mode="c"] f_i not None,
        np.ndarray[int, ndim=3, mode="c"] C_K_ij not None,
        np.ndarray[double, ndim=1, mode="c"] scratch_j not None,
        np.ndarray[double, ndim=3, mode="c"] p_K_ij not None
    ):
    _p_K_ij_equation(
            <double*> np.PyArray_DATA( log_nu_K_i ),
            <double*> np.PyArray_DATA( b_K_i ),
            <double*> np.PyArray_DATA( f_i ),
            <int*> np.PyArray_DATA( C_K_ij ),
            log_nu_K_i.shape[0],
            log_nu_K_i.shape[1],
            <double*> np.PyArray_DATA( scratch_j ),
            <double*> np.PyArray_DATA( p_K_ij )
        )

def f_K_equation(
        np.ndarray[double, ndim=2, mode="c"] b_K_i not None,
        np.ndarray[double, ndim=1, mode="c"] f_i not None,
        np.ndarray[double, ndim=1, mode="c"] scratch_j not None,
        np.ndarray[double, ndim=1, mode="c"] f_K not None
    ):
    _f_K_equation(
            <double*> np.PyArray_DATA( b_K_i ),
            <double*> np.PyArray_DATA( f_i ),
            b_K_i.shape[0],
            b_K_i.shape[1],
            <double*> np.PyArray_DATA( scratch_j ),
            <double*> np.PyArray_DATA( f_K )
        )
