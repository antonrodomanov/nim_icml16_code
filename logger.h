#ifndef LOGGER_H
#define LOGGER_H

#include "LogRegOracle.h"
#include <ctime>

#include <Eigen/Dense>

class Logger {
public:
    Logger(const LogRegOracle& func);

    void log(const Eigen::VectorXd& w);

    std::vector<double> trace_epoch;
    std::vector<double> trace_elaps;
    std::vector<double> trace_val;
    std::vector<double> trace_norm_grad;

private:
    const LogRegOracle& func;
    const int how_often; /* frequency of logging, in number of calls */
    int n_calls;
    clock_t t_start; /* time of creation of this logger */
    double mainten_time; /* total time spent on maintenance tasks */
};

#endif
