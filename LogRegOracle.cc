#include <iostream>

#include "LogRegOracle.h"
#include "special.h"

LogRegOracle::LogRegOracle(const Eigen::MatrixXd& Z, double lambda)
    : Z(Z), lambda(lambda)
{
}

int LogRegOracle::n_samples() const { return Z.rows(); }

double LogRegOracle::single_val(const Eigen::VectorXd& w, int i) const
{
    return logaddexp(0, Z.row(i).dot(w)) + (lambda / 2) * w.squaredNorm();
}

Eigen::VectorXd LogRegOracle::single_grad(const Eigen::VectorXd& w, int i) const
{
    return sigm(Z.row(i).dot(w)) * Z.row(i).transpose() + lambda * w;
}

double LogRegOracle::full_val(const Eigen::VectorXd& w) const
{
    return (1.0 / Z.rows()) * (Z * w).unaryExpr(std::ptr_fun(logaddexp0)).sum() + (lambda / 2) * w.squaredNorm();
}

Eigen::VectorXd LogRegOracle::full_grad(const Eigen::VectorXd& w) const
{
    return (1.0 / Z.rows()) * Z.transpose() * (Z * w).unaryExpr(std::ptr_fun(sigm)) + lambda * w;
}

double LogRegOracle::phi_prime(double mu) const
{
    return sigm(mu);
}

double LogRegOracle::phi_double_prime(double mu) const
{
    double s = sigm(mu);
    return s * (1 - s);
}
