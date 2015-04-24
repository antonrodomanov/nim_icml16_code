#ifndef SPECIAL_H
#define SPECIAL_H

/* Calculate log(exp(a) + exp(b)) without floating-point overflows */
double logaddexp(double a, double b);
double logaddexp0(double x); // the same but for log(1 + exp(x))

/* Sigmoid function */
double sigm(double x);

#endif
