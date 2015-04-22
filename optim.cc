#include "optim.h"
#include <random>

std::vector<double> SGD(const LogRegOracle& func, const std::vector<double>& w0, double alpha, int maxiter)
{
    std::vector<double> w = w0;

    /* prepare random number generator */
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, func.n_samples());

    /* main loop */
    for (int iter = 0; iter < maxiter; ++iter) {
        int idx = dis(gen);

        std::vector<double> gi = func.single_grad(w, idx);

        /* make a step w -= alpha*gi */
        for (int j = 0; j < int(w.size()); ++j) {
            w[j] -= alpha * gi[j];
        }
    }

    return w;
}
