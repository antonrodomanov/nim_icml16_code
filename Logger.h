#ifndef LOGGER_H
#define LOGGER_H

#include <ctime>

#include <Eigen/Dense>

#include "LogRegOracle.h"

class Logger {
public:
    /* Arguments:
        func = the function being traced
        n_logs_per_epoch = the number of requested logs per epoch (the larger this value, the more frequent the logs are)
    */
    Logger(const LogRegOracle& func, double n_logs_per_epoch=10.0);

    void log(const Eigen::VectorXd& w);

    std::vector<double> trace_epoch; // epoch number
    std::vector<double> trace_elaps; // elapsed time
    std::vector<double> trace_val; // function value
    std::vector<double> trace_norm_grad; // gradient norm

private:
    const LogRegOracle& func; // the function being traced
    const int how_often; // frequency of logging, in number of calls

    int n_calls; // number of times this logger has been called
    clock_t t_start; // time of creation of this logger
    double mainten_time; // total time spent on maintenance tasks
};

#endif
