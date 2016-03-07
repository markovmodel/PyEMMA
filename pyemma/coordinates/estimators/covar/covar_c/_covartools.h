/*
# This file is part of PyEMMA.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# PyEMMA is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

void _subtract_row_double(double* X, double* row, int M, int N);
void _subtract_row_float(double* X, double* row, int M, int N);
void _subtract_row_double_copy(double* X0, double* X, double* row, int M, int N);
void _variable_cols_char(int* cols, char* X, int M, int N, int min_constant);
void _variable_cols_int(int* cols, int* X, int M, int N, int min_constant);
void _variable_cols_long(int* cols, long* X, int M, int N, int min_constant);
void _variable_cols_float(int* cols, float* X, int M, int N, int min_constant);
void _variable_cols_double(int* cols, double* X, int M, int N, int min_constant);
void _variable_cols_float_approx(int* cols, float* X, int M, int N, float tol, int min_constant);
void _variable_cols_double_approx(int* cols, double* X, int M, int N, double tol, int min_constant);
