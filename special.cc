#include "special.h"
#include <cmath>
#include <algorithm>

double logaddexp(double a, double b)
{
    double t = std::max(a, b);
    return t + log(exp(a - t) + exp(b - t));
}

double sigm(double x)
{
    return 1.0 / (1 + exp(-x));
}

double infnorm(const std::vector<double>& a)
{
    double t = 0;
    for (int i = 0; i < int(a.size()); ++i) {
        if (fabs(a[i]) > t) {
            t = fabs(a[i]);
        }
    }
    return t;
}
