#include "optim.h"
#include <random>
#include <vector>
#include "logger.h"

Logger SGD(const LogRegOracle& func, const std::vector<double>& w0, double alpha, int maxiter)
{
    Logger logger(func);

    std::vector<double> w = w0;

    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, func.n_samples() - 1);

    /* log the initial point */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        int idx = dis(gen);

        std::vector<double> gi = func.single_grad(w, idx);

        /* make a step w -= alpha*gi */
        for (int j = 0; j < int(w.size()); ++j) {
            w[j] -= alpha * gi[j];
        }

        /* log current point */
        logger.log(w);
    }

    return logger;
}

Logger SAG(const LogRegOracle& func, const std::vector<double>& w0, double alpha, int maxiter)
{
    Logger logger(func);

    std::vector<double> w = w0;

    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, func.n_samples() - 1);

    /* initialise the table of gradients */
    std::vector<std::vector<double>> gtab(func.n_samples(), std::vector<double>(w.size(), 0.0));

    /* initialise average gradient */
    std::vector<double> ag = std::vector<double>(w.size(), 0.0);

    /* log the initial point */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        /* choose index */
        int idx = dis(gen);

        /* compute gradient with chosen index */
        std::vector<double> gi = func.single_grad(w, idx);

        /* modify average gradient: ag += (1/N)*(gi - gtab[idx]) */
        for (int j = 0; j < int(w.size()); ++j) {
            ag[j] += (1.0 / func.n_samples()) * (gi[j] - gtab[idx][j]);
        }

        /* modify table: gtab[idx] = gi */
        for (int j = 0; j < int(w.size()); ++j) {
            gtab[idx][j] = gi[j];
        }

        /* make a step w -= alpha*ag */
        for (int j = 0; j < int(w.size()); ++j) {
            w[j] -= alpha * ag[j];
        }

        /* log current point */
        logger.log(w);
    }

    return logger;
}

/* ============================================================================================================== */
/* ============================================================================================================== */
/* ============================================================================================================== */
/* ============================================================================================================== */
/* ============================================================================================================== */

Logger SO2(const LogRegOracle& func, const std::vector<double>& w0, int maxiter)
{
    Logger logger(func);

    std::vector<double> w = w0;

    /* initialisation */
    std::vector<double> phi_prime(func.n_samples(), 0.0);
    std::vector<double> phi_double_prime(func.n_samples(), 0.0);
    std::vector<double> mu(func.n_samples(), 0.0);

    std::vector<double> g(w.size(), 0.0);
    std::vector<double> p(w.size(), 0.0);

    /* initialise B = lambda^{-1}*I */
    std::vector<std::vector<double>> B(w.size(), std::vector<double>(w.size(), 0.0));
    for (int j = 0; j < int(w.size()); ++j) {
        B[j][j] = 1.0 / func.lambda;
    }

    int idx = -1;

    /* log the initial point */
    logger.log(w);

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        /* choose index */
        idx = (idx + 1) % func.n_samples();

        /* compute mu[i] = z[i]' * v[i] where v[i] = w */
        double mu_new = 0.0;
        for (int j = 0; j < int(w.size()); ++j) {
            mu_new += func.Z[idx][j] * w[j];
        }

        /* compute phi' and phi'' at mui */
        double phi_prime_new = func.phi_prime(mu_new);
        double phi_double_prime_new = func.phi_double_prime(mu_new);

        /* update g */
        double delta_phi_prime = phi_prime_new - phi_prime[idx];
        for (int j = 0; j < int(w.size()); ++j) {
            g[j] += (1.0 / func.n_samples()) * delta_phi_prime * func.Z[idx][j];
        }

        /* update p */
        double delta_phi_double_prime_mu = phi_double_prime_new * mu_new - phi_double_prime[idx] * mu[idx];
        for (int j = 0; j < int(w.size()); ++j) {
            p[j] += (1.0 / func.n_samples()) * delta_phi_double_prime_mu * func.Z[idx][j];
        }

        /* update B */
        double delta_phi_double_prime = phi_double_prime_new - phi_double_prime[idx];
        double coef = (1.0 / func.n_samples()) * delta_phi_double_prime;
        /* calculate bzi = B * z_i */
        std::vector<double> bzi(w.size());
        for (int j1 = 0; j1 < int(w.size()); ++j1) {
            bzi[j1] = 0.0;
            for (int j2 = 0; j2 < int(w.size()); ++j2) {
                bzi[j1] += B[j1][j2] * func.Z[idx][j2];
            }
        }
        /* calculate z_i' * bzi */
        double zi_bzi = 0.0;
        for (int j = 0; j < int(w.size()); ++j) {
            zi_bzi += func.Z[idx][j] * bzi[j];
        }
        /* modify B */
        for (int j1 = 0; j1 < int(w.size()); ++j1) {
            for (int j2 = 0; j2 < int(w.size()); ++j2) {
                B[j1][j2] -= (coef * bzi[j1] * bzi[j2]) / (1 + coef * zi_bzi);
            }
        }

        /* update model */
        mu[idx] = mu_new;
        phi_prime[idx] = phi_prime_new;
        phi_double_prime[idx] = phi_double_prime_new;

        /* calculate direction */
        std::vector<double> d(w.size());
        /* calculate B_gmp = B * (g - p) */
        std::vector<double> B_gmp(w.size());
        for (int j1 = 0; j1 < int(w.size()); ++j1) {
            B_gmp[j1] = 0.0;
            for (int j2 = 0; j2 < int(w.size()); ++j2) {
                B_gmp[j1] += B[j1][j2] * (g[j2] - p[j2]);
            }
        }
        for (int j = 0; j < int(w.size()); ++j) {
            d[j] = w[j] + B_gmp[j];
        }

        /* make a step w -= alpha*ag */
        double alpha = 1.0;
        for (int j = 0; j < int(w.size()); ++j) {
            w[j] -= alpha * d[j];
        }

        /* log current point */
        logger.log(w);
    }

    return logger;
}
