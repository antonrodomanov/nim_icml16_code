#include "LogRegOracle.h"
#include <cmath>

LogRegOracle::LogRegOracle(const std::vector<std::vector<double>>& Z, double lambda)
    : Z(Z), lambda(lambda)
{
}

double LogRegOracle::single_val(const std::vector<double>& w, int idx)
{
    /* compute dot product w' * z[idx] */
    double wtz = 0.0;
    for (int j = 0; j < int(w.size()); ++j) {
        wtz += w[j] * Z[idx][j];
    }

    /* compute squared two-norm */
    double w2 = 0.0;
    for (int j = 0; j < int(w.size()); ++j) {
        w2 += w[j] * w[j];
    }

    return log(1 + exp(wtz)) + (lambda / 2) * w2;
}

std::vector<double> LogRegOracle::single_grad(const std::vector<double>& w, int idx)
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

double LogRegOracle::full_val(const std::vector<double>& w)
{
    double f = 0.0;
    for (int i = 0; i < int(Z.size()); ++i) {
        f += single_val(w, i);
    }
    f /= Z.size();
    return f;
}

std::vector<double> LogRegOracle::full_grad(const std::vector<double>& w)
{
    std::vector<double> g = std::vector<double>(w.size(), 0.0);
    for (int i = 0; i < int(Z.size()); ++i) {
        std::vector<double> gi = single_grad(w, i);
        for (int j = 0; j < int(gi.size()); ++j) {
            g[j] += gi[j];
        }
    }

    /* normalise */
    for (int j = 0; j < int(g.size()); ++j) {
        g[j] /= Z.size();
    }

    return g;
}
