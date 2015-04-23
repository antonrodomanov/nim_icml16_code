#ifndef SPECIAL_H
#define SPECIAL_H

#include <vector>

/* Calculate log(exp(a) + exp(b)) without floating-point overflows */
double logaddexp(double a, double b);

/* Sigmoid function */
double sigm(double x);

/* Calculate infinity norm of vector */
double infnorm(const std::vector<double>& a);

#endif
