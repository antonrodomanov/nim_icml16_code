#include "logger.h"
#include <vector>
#include <cstdio>
#include "special.h"

Logger::Logger(const LogRegOracle& func) : func(func), how_often(func.n_samples() / 10), n_calls(0) {}

void Logger::log(const std::vector<double>& w)
{
    if (n_calls % how_often == 0) {
        double epoch = double(n_calls) / func.n_samples();
        double f = func.full_val(w);
        double norm_g = infnorm(func.full_grad(w));

        fprintf(stderr, "%9.2f %15.6e %15.6e\n", epoch, f, norm_g);

        trace_epoch.push_back(epoch);
        trace_val.push_back(f);
        trace_norm_grad.push_back(norm_g);
    }
    ++n_calls;
}
