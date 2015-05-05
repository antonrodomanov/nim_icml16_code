#ifndef OPTIM_H
#define OPTIM_H

#include <Eigen/Dense>

#include "LogRegOracle.h"
#include "Logger.h"

/* Method SGD */
Eigen::VectorXd SGD(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, size_t maxiter);

/* Method SAG for **linear models** */
Eigen::VectorXd SAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, size_t maxiter);

/* Method SO2 for **linear models** */
Eigen::VectorXd SO2(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter);

#endif
