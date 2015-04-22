#include "special.h"
#include <cmath>
#include <algorithm>

double logaddexp(double a, double b)
{
    double t = std::max(a, b);
    return t + log(exp(a - t) + exp(b - t));
}
