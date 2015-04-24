#ifndef LOG_REG_ORACLE_H_
#define LOG_REG_ORACLE_H_

#include <Eigen/Dense>

class LogRegOracle
{
public:
    LogRegOracle(const Eigen::MatrixXd& Z, double lambda);

    int n_samples() const;

    double single_val(const Eigen::VectorXd& w, int i) const;
    Eigen::VectorXd single_grad(const Eigen::VectorXd& w, int i) const;

    double full_val(const Eigen::VectorXd& w) const;
    Eigen::VectorXd full_grad(const Eigen::VectorXd& w) const;

    double phi_prime(double mu) const;
    double phi_double_prime(double mu) const;

    const Eigen::MatrixXd& Z;
    double lambda;
};

#endif
