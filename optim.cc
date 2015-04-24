#include "optim.h"
#include <random>
#include "logger.h"
#include <iostream>

Eigen::VectorXd SGD(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, int maxiter)
{
    Eigen::VectorXd w = w0;

    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, func.n_samples() - 1);

    /* log the initial point */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        int idx = dis(gen);

        Eigen::VectorXd gi = func.single_grad(w, idx);

        /* make a step w -= alpha*gi */
        w -= alpha * gi;

        /* log current point */
        logger.log(w);
    }

    return w;
}

Eigen::VectorXd SAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, int maxiter)
{
    Eigen::VectorXd w = w0;

    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, func.n_samples() - 1);

    /* initialisation */
    Eigen::VectorXd phi_prime = Eigen::VectorXd::Zero(func.n_samples());

    Eigen::VectorXd g = Eigen::VectorXd::Zero(w.size());

    /* log the initial point */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        /* choose index */
        int idx = dis(gen);

        /* compute mu = z[i]' * w */
        double mu = func.Z.row(idx).dot(w);

        /* compute phi' at mu */
        double phi_prime_new = func.phi_prime(mu);

        /* update g */
        double delta_phi_prime = phi_prime_new - phi_prime(idx);
        g += (1.0 / func.n_samples()) * delta_phi_prime * func.Z.row(idx).transpose();

        /* update model */
        phi_prime(idx) = phi_prime_new;

        /* make a step w -= alpha * (g + lambda * w) */
        w -= alpha * (g + func.lambda * w);

        /* log current point */
        logger.log(w);
    }

    return w;
}

/* ============================================================================================================== */
/* ============================================================================================================== */
/* ============================================================================================================== */
/* ============================================================================================================== */
/* ============================================================================================================== */

Eigen::VectorXd SO2(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, int maxiter)
{
    Eigen::VectorXd w = w0;

    /* initialisation */
    Eigen::VectorXd phi_prime = Eigen::VectorXd::Zero(func.n_samples());
    Eigen::VectorXd phi_double_prime = Eigen::VectorXd::Zero(func.n_samples());
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(func.n_samples());

    Eigen::VectorXd g = Eigen::VectorXd::Zero(w.size());
    Eigen::VectorXd p = Eigen::VectorXd::Zero(w.size());

    /* initialise B = lambda^{-1}*I */
    Eigen::MatrixXd B = (1.0 / func.lambda) * Eigen::MatrixXd::Identity(w.size(), w.size());

    int idx = -1;

    /* log the initial point */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        /* choose index */
        idx = (idx + 1) % func.n_samples();

        /* compute mu[i] = z[i]' * v[i] where v[i] = w */
        double mu_new = func.Z.row(idx).dot(w);

        /* compute phi' and phi'' at mui */
        double phi_prime_new = func.phi_prime(mu_new);
        double phi_double_prime_new = func.phi_double_prime(mu_new);

        /* update g */
        double delta_phi_prime = phi_prime_new - phi_prime(idx);
        g += (1.0 / func.n_samples()) * delta_phi_prime * func.Z.row(idx).transpose();

        /* update p */
        double delta_phi_double_prime_mu = phi_double_prime_new * mu_new - phi_double_prime(idx) * mu(idx);
        p += (1.0 / func.n_samples()) * delta_phi_double_prime_mu * func.Z.row(idx).transpose();

        /* update B */
        double delta_phi_double_prime = phi_double_prime_new - phi_double_prime(idx);
        double coef = (1.0 / func.n_samples()) * delta_phi_double_prime;
        /* calculate bzi = B * z_i */
        Eigen::VectorXd bzi = B * func.Z.row(idx).transpose();
        /* calculate z_i' * bzi */
        double zi_bzi = func.Z.row(idx).dot(bzi);
        /* modify B */
        B -= (coef / (1 + coef * zi_bzi)) * bzi * bzi.transpose();

        /* update model */
        mu(idx) = mu_new;
        phi_prime(idx) = phi_prime_new;
        phi_double_prime(idx) = phi_double_prime_new;

        /* calculate direction d = -(w + B * (g - p)) */
        Eigen::VectorXd d = -(w + B * (g - p));

        /* make a step w += alpha*d */
        double alpha = 1.0;
        w += alpha * d;

        /* log current point */
        logger.log(w);
    }

    return w;
}
