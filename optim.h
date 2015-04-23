#ifndef OPTIM_H
#define OPTIM_H

#include "LogRegOracle.h"
#include "logger.h"

Logger SGD(const LogRegOracle& func, const std::vector<double>& w0, double alpha, int maxiter);

#endif
