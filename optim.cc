#include <iostream>
#include <random>

#include "optim.h"
#include "Logger.h"

/* ****************************************************************************************************************** */
/* *************************************************** SGD ********************************************************** */
/* ****************************************************************************************************************** */
Eigen::VectorXd SGD(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, int maxiter)
{
    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, func.n_samples() - 1);

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        /* select random sample i */
        int i = dis(gen);

        /* compute its gradient g_i = nabla f_i(w) */
        Eigen::VectorXd gi = func.single_grad(w, i);

        /* make a step w -= alpha * g_i */
        w -= alpha * gi;

        /* log current position */
        logger.log(w);
    }

    return w;
}

/* ****************************************************************************************************************** */
/* *************************************************** SAG ********************************************************** */
/* ****************************************************************************************************************** */
Eigen::VectorXd SAG(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, double alpha, int maxiter)
{
    /* assign useful variables */
    const int N = func.n_samples();
    const int D = w0.size();
    const double lambda = func.lambda;
    const Eigen::MatrixXd& Z = func.Z;

    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, N - 1);

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* initialisation */
    Eigen::VectorXd phi_prime = Eigen::VectorXd::Zero(N); // coefficients phi_prime(i) = phi'(z_i' * v_i)

    Eigen::VectorXd g = Eigen::VectorXd::Zero(D); // average gradient g = 1/N sum_i nabla f_i(v_i)

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        /* select random sample i */
        int i = dis(gen);

        /* take i-th training sample */
        Eigen::VectorXd zi = Z.row(i).transpose();

        /* compute phi_prime_new = phi'(z_i' * w) */
        double phi_prime_new = func.phi_prime(zi.dot(w));

        /* update g: g += 1/N delta_phi_prime * z_i */
        double delta_phi_prime = phi_prime_new - phi_prime(i);
        g += (1.0 / N) * delta_phi_prime * zi;

        /* update model */
        phi_prime(i) = phi_prime_new;

        /* make a step w -= alpha * (g + lambda * w) */
        w -= alpha * (g + lambda * w);

        /* log current position */
        logger.log(w);
    }

    return w;
}

/* ****************************************************************************************************************** */
/* *************************************************** SO2 ********************************************************** */
/* ****************************************************************************************************************** */
Eigen::VectorXd SO2(const LogRegOracle& func, Logger& logger, const Eigen::VectorXd& w0, int maxiter)
{
    /* assign useful variables */
    const int N = func.n_samples();
    const int D = w0.size();
    const double lambda = func.lambda;
    const Eigen::MatrixXd& Z = func.Z;

    /* assign starting point */
    Eigen::VectorXd w = w0;

    /* initialisation */
    Eigen::VectorXd mu = Eigen::VectorXd::Zero(N); // coefficients mu_i = z_i' * v_i
    Eigen::VectorXd phi_prime = Eigen::VectorXd::Zero(N); // coefficients phi_prime(i) = phi'(mu_i)
    Eigen::VectorXd phi_double_prime = Eigen::VectorXd::Zero(N); // coefficients phi_doube_prime(i) = phi''(mu_i)

    Eigen::VectorXd g = Eigen::VectorXd::Zero(D); // average gradient g = 1/N sum_i nabla f_i(v_i)
    Eigen::VectorXd p = Eigen::VectorXd::Zero(D); // vector p = 1/N sum_i nabla^2 f_i(v_i) v_i

    Eigen::MatrixXd B = (1.0 / lambda) * Eigen::MatrixXd::Identity(D, D); // inverse average hessian B = (1/N sum_i nabla^2 f_i(v_i))^{-1}

    int i = -1; // sample index; start with -1 because the first one will be (i+1) % N = 0
    double alpha = 1.0; // step length (always use unit step length for now)

    /* log initial position */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        /* choose index; use cyclic order */
        i = (i + 1) % N;

        /* take i-th training sample */
        Eigen::VectorXd zi = Z.row(i).transpose();

        /* compute new mu_i = z_i' * v_i where v_i = w */
        double mu_new = zi.dot(w);

        /* compute phi' and phi'' at mu_i */
        double phi_prime_new = func.phi_prime(mu_new);
        double phi_double_prime_new = func.phi_double_prime(mu_new);

        /* update g: g += 1/N delta_phi_prime z_i */
        double delta_phi_prime = phi_prime_new - phi_prime(i);
        g += (1.0 / N) * delta_phi_prime * zi;

        /* update p: p += 1/N (phi_double_prime_new * mu_new - phi_double_prime * mu) * z_i */
        double delta_phi_double_prime_mu = phi_double_prime_new * mu_new - phi_double_prime(i) * mu(i);
        p += (1.0 / N) * delta_phi_double_prime_mu * zi;

        /* update B using Sherman-Morrison-Woodbury formula (rank-1 update) */
        double delta_phi_double_prime = phi_double_prime_new - phi_double_prime(i);
        double coef = (1.0 / N) * delta_phi_double_prime;
        Eigen::VectorXd bzi = B * zi;
        B.noalias() -= (coef / (1.0 + coef * zi.dot(bzi))) * bzi * bzi.transpose();

        /* update model */
        mu(i) = mu_new;
        phi_prime(i) = phi_prime_new;
        phi_double_prime(i) = phi_double_prime_new;

        /* calculate direction d = -(w + B * (g - p)) */
        Eigen::VectorXd d = -(w + B * (g - p));

        /* make a step w += alpha * d */
        w += alpha * d;

        /* log current position */
        logger.log(w);
    }

    return w;
}
