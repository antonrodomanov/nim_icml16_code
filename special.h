#ifndef SPECIAL_H
#define SPECIAL_H

#include <vector>

/* Calculate log(exp(a) + exp(b)) without floating-point overflows */
double logaddexp(double a, double b);

/* Sigmoid function */
double sigm(double x);

#endif
