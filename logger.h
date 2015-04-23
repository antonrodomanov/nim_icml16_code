#ifndef LOGGER_H
#define LOGGER_H

#include "LogRegOracle.h"
#include <vector>

class Logger {
public:
    Logger(const LogRegOracle& func);

    void log(const std::vector<double>& w);

    std::vector<double> trace_epoch;
    std::vector<double> trace_val;

private:
    const LogRegOracle& func;
    const int how_often; /* frequency of logging, in number of calls */
    int n_calls;
};

#endif
