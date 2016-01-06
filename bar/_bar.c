/*
* This file is part of thermotools.
*
* Copyright 2015 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
*
* thermotools is free software: you can redistribute it and/or modify
* it under the terms of the GNU Lesser General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "../util/_util.h"

extern double _bar_df(double *db_IJ, int L1, double *db_JI, int L2, double *scratch)
{
    int i;
    double ln_avg1;
    double ln_avg2; 
    for (i=0; i<L1; i++)
    {
        scratch[i] = db_IJ[i]>0 ? 0 : db_IJ[i];
    }
    ln_avg1 = _logsumexp_sort_kahan_inplace(scratch, L1);
    for (i=0; i<L1; i++)
    {
        scratch[i] = db_JI[i]>0 ? 0 : db_JI[i];
    }
    ln_avg2 = _logsumexp_sort_kahan_inplace(scratch, L2);
    return ln_avg2 - ln_avg1;
}