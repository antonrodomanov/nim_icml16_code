#ifndef OPTIM_H
#define OPTIM_H

#include "LogRegOracle.h"
#include "logger.h"

#include <Eigen/Dense>

Eigen::VectorXd SGD(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, int maxiter);

Eigen::VectorXd SAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, int maxiter);

Eigen::VectorXd SO2(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, int maxiter);

#endif
