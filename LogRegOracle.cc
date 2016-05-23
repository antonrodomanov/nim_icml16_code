#include <iostream>

#include "LogRegOracle.h"
#include "special.h"

LogRegOracle::LogRegOracle(const Eigen::MatrixXd& Z, double lambda, double lambda1, int minibatch_size)
    : CompositeFunction(lambda1), Z(Z), lambda(lambda)
{
    /* Auxiliary variables */
    const int n = Z.rows();
    const int d = Z.cols();

    /* Construct a random permutation */
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::random_shuffle(perm.begin(), perm.end());

    /* Split training samples into list of matrices of size ~`minibatch_size` */
    n_minibatches = ceil(double(n) / minibatch_size);
    minibatch_sizes.resize(n_minibatches);
    int i = 0;
    int size_rem = n % minibatch_size; // size of the "last" minibatch
    for (int j = 0; j < n_minibatches; ++j) {
        // Determine the size of this minibatch
        minibatch_sizes[j] = (size_rem == 0 || j < n_minibatches - 1) ? minibatch_size : size_rem;
        Eigen::MatrixXd Z_minibatch(minibatch_sizes[j], d);
        for (int k = 0; k < minibatch_sizes[j]; ++k) {
            Z_minibatch.row(k) = Z.row(perm[i++]);
        }
        Z_list.emplace_back(Z_minibatch);
    }
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

double LogRegOracle::full_val_grad(const Eigen::VectorXd& w, Eigen::VectorXd& g) const
{
    Eigen::VectorXd zw = Z * w;
    g = (1.0 / Z.rows()) * Z.transpose() * zw.unaryExpr(std::ptr_fun(sigm)) + lambda * w;
    return (1.0 / Z.rows()) * zw.unaryExpr(std::ptr_fun(logaddexp0)).sum() + (lambda / 2) * w.squaredNorm();
}

LogRegHessVec LogRegOracle::hessvec() const
{
    return LogRegHessVec(Z, lambda);
}

Eigen::VectorXd LogRegOracle::phi_prime(const Eigen::VectorXd& mu) const
{
    return mu.unaryExpr(std::ptr_fun(sigm));
}

Eigen::VectorXd LogRegOracle::phi_double_prime(const Eigen::VectorXd& mu) const
{
    Eigen::VectorXd s = mu.unaryExpr(std::ptr_fun(sigm));
    return s.array() * (1 - s.array());
}

/* ****************************************************************************************************************** */
/* ************************************************ LogRegHessVec *************************************************** */
/* ****************************************************************************************************************** */

LogRegHessVec::LogRegHessVec(const Eigen::MatrixXd& Z, double lambda)
    : Z(Z), lambda(lambda)
{
}

void LogRegHessVec::prepare(const Eigen::VectorXd& w)
{
    Eigen::VectorXd sigma = (Z * w).unaryExpr(std::ptr_fun(sigm));
    s = sigma.array() * (1.0 - sigma.array());
}

Eigen::VectorXd LogRegHessVec::calculate(const Eigen::VectorXd& d) const
{
    return (1.0 / Z.rows()) * Z.transpose() * (s.array() * (Z * d).array()).matrix() + lambda * d;
}
