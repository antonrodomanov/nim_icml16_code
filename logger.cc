#include "logger.h"
#include <vector>
#include <cstdio>

Logger::Logger(const LogRegOracle& func) : func(func), how_often(func.n_samples() / 10), n_calls(0) {}

void Logger::log(const std::vector<double>& w)
{
    if (n_calls % how_often == 0) {
        trace_epoch.push_back(double(n_calls) / func.n_samples());
        trace_val.push_back(func.full_val(w));
    }
    ++n_calls;
}
