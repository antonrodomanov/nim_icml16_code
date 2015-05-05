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

Eigen::MatrixXd LogRegOracle::full_hess(const Eigen::VectorXd& w) const
{
    /* calcuate the diagonal part */
    Eigen::VectorXd sigma = (Z * w).unaryExpr(std::ptr_fun(sigm));
    Eigen::VectorXd s = sigma.array() * (1.0 - sigma.array());

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(w.size(), w.size());
    /* since we don't want to copy the entire Z matrix, the only way to compute Z.T*S*Z
       without allocating new memory is to loop over all the samples */
    for (int i = 0; i < n_samples(); ++i) {
        H.selfadjointView<Eigen::Upper>().rankUpdate(Z.row(i).transpose(), s(i));
    }
    //Eigen::MatrixXd ZS = Z.array().colwise() * s.unaryExpr(std::ptr_fun(sqrt)).array();
    //H.selfadjointView<Eigen::Upper>().rankUpdate(ZS.transpose());
    H /= Z.rows(); // normalise by the number of samples

    /* don't forget the regulariser */
    H += lambda * Eigen::MatrixXd::Identity(w.size(), w.size());

    return H;
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
