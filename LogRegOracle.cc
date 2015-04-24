#include "LogRegOracle.h"
#include <cmath>
#include "special.h"
#include <iostream>

LogRegOracle::LogRegOracle(const Eigen::MatrixXd& Z, double lambda)
    : Z(Z), lambda(lambda)
{
}

int LogRegOracle::n_samples() const { return Z.rows(); }

double LogRegOracle::single_val(const Eigen::VectorXd& w, int idx) const
{
    /* compute dot product w' * z[idx] */
    double wtz = Z.row(idx).dot(w);

    /* compute squared two-norm */
    double w2 = w.squaredNorm();

    return logaddexp(0, wtz) + (lambda / 2) * w2;
}

Eigen::VectorXd LogRegOracle::single_grad(const Eigen::VectorXd& w, int idx) const
{
    /* compute dot product w' * z[idx] */
    double wtz = Z.row(idx).dot(w);

    /* take sigmoid */
    double s = sigm(wtz);

    /* compute requested gradient: g = s * z[idx] + lambda * w */
    Eigen::VectorXd g = s * Z.row(idx).transpose() + lambda * w;

    return g;
}

double LogRegOracle::full_val(const Eigen::VectorXd& w) const
{
    Eigen::VectorXd zw = Z * w;

    double f;
    f = zw.unaryExpr(std::ptr_fun(logaddexp0)).sum();
    f /= Z.rows();
    f += (lambda / 2) * w.squaredNorm();
    return f;
}

Eigen::VectorXd LogRegOracle::full_grad(const Eigen::VectorXd& w) const
{
    Eigen::VectorXd g = Eigen::VectorXd::Zero(w.size());
    for (int i = 0; i < Z.rows(); ++i) {
        g += single_grad(w, i);
    }

    /* normalise */
    g /= Z.rows();

    return g;
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
