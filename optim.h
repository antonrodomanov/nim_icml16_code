#ifndef OPTIM_H
#define OPTIM_H

#include "LogRegOracle.h"
#include "logger.h"

#include <Eigen/Dense>

Logger SGD(const LogRegOracle& func, const Eigen::VectorXd& w0, double alpha, int maxiter);

Logger SAG(const LogRegOracle& func, const Eigen::VectorXd& w0, double alpha, int maxiter);

Logger SO2(const LogRegOracle& func, const Eigen::VectorXd& w0, int maxiter);

#endif
