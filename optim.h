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
Eigen::VectorXd SO2(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double alpha);

/* Newton's method for a general strongly convex function */
Eigen::VectorXd newton(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double c1=1e-4);

/* Hessian-free Newton method */
Eigen::VectorXd HFN(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double c1=1e-4);

/* Method BFGS for a general strongly convex function */
Eigen::VectorXd BFGS(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, double c1=1e-4);

/* Method L-BFGS for a general strongly convex function */
Eigen::VectorXd LBFGS(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, size_t maxiter, size_t m=10, double c1=1e-4);

#endif
