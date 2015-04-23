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
