#include "LogRegOracle.h"
#include <cmath>
#include "special.h"

LogRegOracle::LogRegOracle(const std::vector<std::vector<double>>& Z, double lambda)
    : Z(Z), lambda(lambda)
{
}

int LogRegOracle::n_samples() const { return Z.size(); }

double LogRegOracle::single_val(const std::vector<double>& w, int idx) const
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

    return logaddexp(0, wtz) + (lambda / 2) * w2;
}

std::vector<double> LogRegOracle::single_grad(const std::vector<double>& w, int idx) const
{
    /* compute dot product w' * z[idx] */
    double wtz = 0.0;
    for (int j = 0; j < int(w.size()); ++j) {
        wtz += w[j] * Z[idx][j];
    }

    /* take sigmoid */
    double s = sigm(wtz);

    /* compute requested gradient */
    std::vector<double> g = std::vector<double>(int(w.size()), 0.0);
    for (int j = 0; j < int(w.size()); ++j) {
        g[j] = s * Z[idx][j];
        /* add reguliriser */
        g[j] += lambda * w[j];
    }

    return g;
}

double LogRegOracle::full_val(const std::vector<double>& w) const
{
    double f = 0.0;
    for (int i = 0; i < int(Z.size()); ++i) {
        f += single_val(w, i);
    }
    f /= Z.size();
    return f;
}

std::vector<double> LogRegOracle::full_grad(const std::vector<double>& w) const
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

double LogRegOracle::phi_prime(double mu) const
{
    return sigm(mu);
}

double LogRegOracle::phi_double_prime(double mu) const
{
    double s = sigm(mu);
    return s * (1 - s);
}
