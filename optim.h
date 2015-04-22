#ifndef OPTIM_H
#define OPTIM_H

#include "LogRegOracle.h"

std::vector<double> SGD(const LogRegOracle& func, const std::vector<double>& w0, double alpha, int maxiter);

#endif
