#include "LogRegOracle.h"
#include <cmath>

LogRegOracle::LogRegOracle(const std::vector<std::vector<double>>& Z, double lambda)
    : Z(Z), lambda(lambda)
{
}

std::vector<double> LogRegOracle::single_grad(std::vector<double> w, int idx)
{
    /* compute dot product w' * z[idx] */
    double wtz = 0.0;
    for (int j = 0; j < int(w.size()); ++j) {
        wtz += w[j] * Z[idx][j];
    }

    /* take sigmoid */
    double sigm = 1.0 / (1 + exp(-wtz));

    /* compute requested gradient */
    std::vector<double> g = std::vector<double>(int(w.size()), 0.0);
    for (int j = 0; j < int(w.size()); ++j) {
        g[j] = sigm * Z[idx][j];
        /* add reguliriser */
        g[j] += lambda * w[j];
    }

    return g;
}
