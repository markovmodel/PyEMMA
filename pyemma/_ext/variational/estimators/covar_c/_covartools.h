#ifndef _covartools_h_
#define _covartools_h_

void _subtract_row_double(double* X, double* row, int M, int N);
void _subtract_row_float(double* X, double* row, int M, int N);
void _subtract_row_double_copy(double* X0, double* X, double* row, int M, int N);
int* _bool_to_list(int* b, int N, int nnz);
void _variable_cols_char(int* cols, char* X, int M, int N, int min_constant);
void _variable_cols_int(int* cols, int* X, int M, int N, int min_constant);
void _variable_cols_long(int* cols, long* X, int M, int N, int min_constant);
void _variable_cols_float(int* cols, float* X, int M, int N, int min_constant);
void _variable_cols_double(int* cols, double* X, int M, int N, int min_constant);
void _variable_cols_float_approx(int* cols, float* X, int M, int N, float tol, int min_constant);
void _variable_cols_double_approx(int* cols, double* X, int M, int N, double tol, int min_constant);

#endif
