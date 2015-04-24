#include <iostream>
#include <vector>
#include <cstdio>
#include <ctime>

#include "logger.h"
#include "special.h"

Logger::Logger(const LogRegOracle& func, double n_logs_per_epoch) :
    func(func),
    how_often(func.n_samples() / n_logs_per_epoch),
    n_calls(0),
    t_start(clock()),
    mainten_time(0.0)
{}

void Logger::log(const Eigen::VectorXd& w)
{
    if (n_calls % how_often == 0) { /* perform actual logging only every `how_often` calls */
        /* remember the time when the maintenance starts */
        clock_t mainten_start = clock();

        /* calculate the number of epoch and optimiser's time */
        double epoch = double(n_calls) / func.n_samples(); // current epoch
        double tot_elaps = double(clock() - t_start) / CLOCKS_PER_SEC; // total time elapsed since optimiser was run
        double opt_elaps = tot_elaps - mainten_time; // `tot_elaps` equals optimiser's time + maintenance time, so
                                                     // subtract maintenance time to get optimiser's time
        /* always start `opt_elaps` at zero (first call is for free) */
        if (n_calls == 0) {
            opt_elaps = 0.0;
        }

        /* calculate function value and gradient norm */
        double f = func.full_val(w);
        double norm_g = func.full_grad(w).lpNorm<Eigen::Infinity>();

        /* print calculated values */
        if (n_calls == 0) { /* print table header when it's the first call */
            fprintf(stderr, "%9s %9s %9s %15s %15s\n", "epoch", "opt_elaps", "tot_elaps", "f", "norm_g");
        }
        fprintf(stderr, "%9.2f %9.2f %9.2f %15.6e %15.6e\n", epoch, opt_elaps, tot_elaps, f, norm_g);

        /* append calculated values to trace */
        trace_epoch.push_back(epoch);
        trace_elaps.push_back(opt_elaps);
        trace_val.push_back(f);
        trace_norm_grad.push_back(norm_g);

        /* increase total maintenance time by the time spent on running this chunk of code */
        mainten_time += double(clock() - mainten_start) / CLOCKS_PER_SEC;
    }

    /* increase total number of calls for this logger */
    ++n_calls;
}
