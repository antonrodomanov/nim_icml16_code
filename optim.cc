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
