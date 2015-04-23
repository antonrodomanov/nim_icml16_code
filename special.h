#ifndef SPECIAL_H
#define SPECIAL_H

#include <vector>

/* Calculate log(exp(a) + exp(b)) without floating-point overflows */
double logaddexp(double a, double b);

/* Calculate infinity norm of vector */
double infnorm(const std::vector<double>& a);

#endif
