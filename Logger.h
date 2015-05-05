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
        tol = if gradient norm at current point below this value, terminate the optimiser
        opt_allowed_time = maximal amount of time for which the optimiser is allowed to work (set -1 for no limit)
    */
    Logger(const LogRegOracle& func, double n_logs_per_epoch=10.0, double tol=1e-5, double opt_allowed_time=-1.0);

    /* Argument `n_full_calls` is used by **non-incremental** optimisers and equals the total number of oracle calls made by the optimiser.
       This function usually returns true; when returns false, the optimiser should terminate.
    */
    bool log(const Eigen::VectorXd& w, size_t n_full_calls=0);

    std::vector<double> trace_epoch; // epoch number
    std::vector<double> trace_elaps; // elapsed time
    std::vector<double> trace_val; // function value
    std::vector<double> trace_norm_grad; // gradient norm

private:
    const LogRegOracle& func; // the function being traced
    const size_t how_often; // frequency of logging, in number of calls

    size_t n_calls; // number of times this logger has been called
    clock_t t_start; // time of creation of this logger
    double mainten_time; // total time spent on maintenance tasks

    double tol; /* gradient norm tolerance */
    double opt_allowed_time; /* allowed time for optimiser */
};

#endif
