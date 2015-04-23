#ifndef OPTIM_H
#define OPTIM_H

#include "LogRegOracle.h"
#include "logger.h"

Logger SGD(const LogRegOracle& func, const std::vector<double>& w0, double alpha, int maxiter);

Logger SAG(const LogRegOracle& func, const std::vector<double>& w0, double alpha, int maxiter);

Logger SO2(const LogRegOracle& func, const std::vector<double>& w0, int maxiter);

#endif
