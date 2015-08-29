/*

    lse.h - logsumexp implementation in C (header file)

    author: Christoph Wehmeyer <christoph.wehmeyer@fu-berlin.de>

*/

#ifndef PYTRAM_LSE
#define PYTRAM_LSE

#include <stdio.h>
#include <math.h>

/* _sort()is based on examples from http://www.linux-related.de (2004) */
void _sort( double *array, int L, int R );

double _logsumexp( double *array, int length );
double _logsumexp_pair( double a, double b );

#endif
